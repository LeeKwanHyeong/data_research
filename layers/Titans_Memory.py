import torch
import torch.nn as nn
import torch.nn.functional as F

# Local Memory Matching (LMM) 모듈
# - 인코더 은닉 벡터(encoded)와 메모리 벡터(memory) 간의 유사도를 계산
# - 각 시점별로 가장 관련성이 높은 top-k 메모리를 선택하여 평균 결합
# - 최종적으로 encoded + matched 메모리 벡터를 선형 변환하여 출력
class LMM(nn.Module):
    def __init__(self, d_model, top_k = 5):
        super().__init__()
        self.top_k = top_k # 각 시점별로 선택할 메모리 슬롯 개수 (k-NN 개념)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, encoded, memory):
        # 입력과 메모리 벡터 정규화 (코사인 유사도 계산을 위해)
        # - encoded: [batch, seq_len, d_model]
        # - memory:  [batch, mem_len, d_model]
        encoded_norm = F.normalize(encoded, dim = -1)
        memory_norm = F.normalize(memory, dim = -1)

        # 유사도 행렬 계산 (코사인 유사도)
        # - similarity: [batch, seq_len, mem_len]
        similarity = torch.matmul(encoded_norm, memory_norm.transpose(-2, -1))

        # 각 시점별로 top-k 메모리 선택
        # - top_k_values: [batch, seq_len, top_k] (유사도 값)
        # - top_k_indices: [batch, seq_len, top_k] (선택된 메모리 인덱스)
        top_k_values, top_k_indices = torch.topk(similarity, self.top_k, dim = -1)

        # 선택된 메모리 인덱스에 따라 메모리 벡터 추출
        B, L, M = top_k_indices.size() # batch, seq_len, top_k
        memory_exp = memory.unsqueeze(1).expand(-1, L, -1, -1) # [B, L, mem_len, d_model]
        top_k_indices_exp = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, memory.size(-1))

        # gather로 top-k 메모리 평균 -> 매칭된 메모리 벡터 생성
        # matched: [B, L, d_model]
        selected = torch.gather(memory_exp, 2, top_k_indices_exp)

        # 입력(encoded)과 matched 메모리를 결합 후 선형 변환
        # output: [B, L, d_model]
        matched = selected.mean(dim = 2)
        output = self.linear(encoded + matched)
        return output

# Efficient Multi-Head Attention with Memory
# - Titan의 MAC(Memory-as-Context) 핵심 모듈
# - 기존 Multi-Head Attention에 두 종류의 메모리 결합:
#   1) Contextual Memory → 최근 시퀀스 기반 단기 메모리 (batch별/슬라이딩 윈도우별 업데이트)
#   2) Persistent Memory → 장기 지식을 저장하는 고정 파라미터 메모리
# - Q(query)는 입력 토큰에서만 생성, K(key)/V(value)는 [Contextual + Persistent + Input] 전체에서 생성
class MemoryAttention(nn.Module):
    def __init__(self, d_model, n_heads, contextual_mem_size, persistent_mem_size):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0 # head_dim이 정수로 나누어 떨어져야 함

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output Projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Persistent Memory (Long-term knowledge)
        self.persistent_memory = nn.Parameter(torch.randn(contextual_mem_size, d_model))

        # Contextual Memory buffer (updated per batch or sliding window)
        self.contextual_mem_size = contextual_mem_size
        self.contextual_memory = None # Initialized Dynamically

    def update_contextual_memory(self, new_context):
        """
                새로운 컨텍스트 벡터를 Contextual Memory에 추가 (FIFO 방식)
                - new_context: (batch_size, context_len, d_model) 또는 (context_len, d_model)
        """
        device = self.persistent_memory.device  # 모델 파라미터 기준 디바이스
        new_context = new_context.detach().to(device)
        flat_context = new_context.view(-1, new_context.size(-1))
        if self.contextual_memory is None:
            # 초기화 시점
            self.contextual_memory = flat_context.to(device)
        else:
            # 기존 메모리에 새 컨텍스트 추가
            self.contextual_memory = torch.cat([self.contextual_memory, flat_context], dim = 0)
            # 최대 크기 초과 시 오래된 메모리 삭제
            if self.contextual_memory.size(0) > self.contextual_mem_size:
                self.contextual_memory = self.contextual_memory[-self.contextual_mem_size:]


    def forward(self, x):
        '''
            x: (B, L, d_model) (Core input tokens)
        '''
        B, L, _ = x.size()

        # Build extended input: [contextual memory + persistent memory + input]
        persistent_mem_expanded = self.persistent_memory.unsqueeze(0).expand(B, -1, -1) # (B, P, d_model)
        contextual_mem = (
            self.contextual_memory.unsqueeze(0).expand(B, -1, -1)
            if self.contextual_memory is None
            else torch.empty(B, 0, self.d_model, device=x.device)
        )
        extended_input = torch.cat([contextual_mem, persistent_mem_expanded, x], dim = 1) # (B, C + P + L, d_model)



        # Q from input only, K/V from extended input
        Q = self.W_q(x)
        K = self.W_k(extended_input)
        V = self.W_v(extended_input)

        # Reshape for multi-head
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, L, D)
        K = K.view(B, extended_input.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, extended_input.size(1), self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim = -1)
        context = torch.matmul(attn_probs, V) # (B, H, L, D)

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(context)


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))


class TitanBackbone(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, contextual_mem_size, persistent_mem_size, dropout = 0.1):
        super().__init__()
        self.attn = MemoryAttention(d_model, n_heads, contextual_mem_size, persistent_mem_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout = dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# Memory Encoder (Core Network)
class MemoryEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, d_ff, contextual_mem_size, persistent_mem_size, dropout = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            TitanBackbone(d_model, n_heads, d_ff, contextual_mem_size, persistent_mem_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return x

