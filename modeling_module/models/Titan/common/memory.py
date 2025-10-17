import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------
# PositionWise Feed-Forward Network (FFN)
# - 각 시점(time step)별로 독립적인 FFN 적용
# - 구조: Linear → SiLU → Dropout → Linear
# -------------------------------------------------------------
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        """
        d_model : 입력 및 출력 차원
        d_ff    : 내부 확장 차원 (보통 4배)
        dropout : 드롭아웃 비율
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)     # 첫 번째 선형 변환 (확장)
        self.linear2 = nn.Linear(d_ff, d_model)     # 두 번째 선형 변환 (축소)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, L, D] - 배치, 시퀀스 길이, 차원
        return: [B, L, D] - same shape
        """
        x = self.linear1(x) # [B, L, d_ff]
        x = F.silu(x)       # [B, L, d_ff] - 활성화 함수 (Swish 계열)
        x = self.dropout(x) # dropout
        x = self.linear2(x) # [B, L, d_model]
        return x

class LMM(nn.Module):
    def __init__(self, d_model: int, top_k: int = 5):
        """
        Local Memory Matching 모듈

        Args:
            d_model: feature 차원 (임베딩 차원)
            top_k: 메모리 내에서 상위 k개의 유사한 메모리 선택
        """
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.scale = 1.0 / (d_model ** 0.5) # 현재는 미사용. scaled dot-product용

    def forward(self, encoded: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        encoded: [B, L, D] - 인코더의 출력 (context representation)
        memory : [B, M, D] or [M, D] - 저장된 메모리 벡터
        return : [B, L, D] - LMM으로 강화된 인코더 출력

        - B: batch size
        - L: 시퀀스 길이
        - M: 메모리의 시퀀스 길이
        - D: feature 차원
        """
        B, L, D = encoded.shape

        # 메모리가 비어있거나 None이면 입력 그대로 반환
        if memory is None or memory.numel() == 0:
            return encoded  # memory가 비어있으면 그대로 통과(안전)

        # [M, D] -> [B, M, D]로 broadcast
        if memory.dim() == 2:  # [M, D] -> [B, M, D]
            memory = memory.unsqueeze(0).expand(B, -1, -1)

        M = memory.size(1)
        if M == 0:
            return encoded  # memory가 비어있으면 그대로 통과

        # top_k를 memory 길이 M에 맞게 제한
        k = min(self.top_k, M)

        # --- 정규화 (cosine similarity용) ---
        encoded_norm = F.normalize(encoded, p=2, dim=-1)    # [B, L, D]
        memory_norm  = F.normalize(memory,  p=2, dim=-1)    # [B, M, D]

        # --- Cosine Similarity 계산 ---
        similarity   = torch.matmul(encoded_norm, memory_norm.transpose(-2, -1))  # [B, L, M]

        # --- top-k 유사 메모리 인덱스 추출 ---
        top_k_vals, top_k_idx = torch.topk(similarity, k, dim=-1)  # [B, L, k], [B, L, k]

        # --- 해당 인덱스를 기반으로 메모리 벡터 gathering ---
        memory_exp     = memory.unsqueeze(1).expand(-1, L, -1, -1)              # [B, L, M, D]
        top_idx_exp    = top_k_idx.unsqueeze(-1).expand(-1, -1, -1, D)          # [B, L, k, D]
        selected_mem   = torch.gather(memory_exp, 2, top_idx_exp)               # [B, L, k, D]

        # --- 평균 내기 (top_k memory vector 평균) ---
        matched        = selected_mem.mean(dim=2)                                # [B, L, D]

        # 간단한 residual 결합(필요시 가중치/게이팅 추가 가능)
        enhanced = encoded + matched
        return enhanced


# -------------------------------------------------------------
# MemoryAttention (Titan의 Memory-as-Context 핵심 모듈)
# -------------------------------------------------------------
class MemoryAttention(nn.Module):
    """
    d_model: 전체 임베딩 차원
    n_heads: multi-head 수
    contextual_mem_size: context 기반 메모리 최대 저장 개수
    persistent_mem_size: 파라미터로 학습되는 장기 메모리 크기
    """
    def __init__(self, d_model, n_heads, contextual_mem_size, persistent_mem_size):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0   # head 수로 나누어 떨어져야 함

        # --- Q, K, V를 위한 Linear Projection ---
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # --- Persistent Memory (학습 가능한 파라미터) ---
        self.persistent_memory = nn.Parameter(torch.empty(persistent_mem_size, d_model))
        nn.init.xavier_uniform_(self.persistent_memory)

        # --- Contextual Memory (최근 시퀀스 저장용 buffer) ---
        self.contextual_mem_size = contextual_mem_size
        self.contextual_memory = None

    def update_contextual_memory(self, new_context):
        """
        슬라이딩 윈도우 방식으로 contextual memory 업데이트
        new_context: [B, L, D] or [L, D]
        """
        device = self.persistent_memory.device
        new_context = new_context.detach().to(device)
        flat_context = new_context.view(-1, new_context.size(-1))

        if torch.isnan(flat_context).any():
            print("[Warning] Skipping update: NaN in new_context")
            return

        # clipping for stability
        flat_context = flat_context.clamp(min=-1e3, max=1e3)

        # 메모리 저장
        if self.contextual_memory is None:
            self.contextual_memory = flat_context
        else:
            self.contextual_memory = torch.cat([self.contextual_memory, flat_context], dim=0)
            if self.contextual_memory.size(0) > self.contextual_mem_size:
                self.contextual_memory = self.contextual_memory[-self.contextual_mem_size:]

    def forward(self, x):
        """
        x: [B, L, D] - 입력 시퀀스
        return: [B, L, D] - memory attention이 적용된 출력
        """
        B, L, _ = x.size()

        # ------------------------------
        # Memory 구성
        # ------------------------------
        persistent_mem = self.persistent_memory.unsqueeze(0).expand(B, -1, -1)  # [B, M_p, D]
        contextual_mem = (
            torch.empty(B, 0, self.d_model, device=x.device)
            if self.contextual_memory is None
            else self.contextual_memory.unsqueeze(0).expand(B, -1, -1)          # [B, M_c, D]
        )
        extended_input = torch.cat([contextual_mem, persistent_mem, x], dim=1)  # [B, M_c + M_p + L, D]

        # ------------------------------
        # Q/K/V 계산
        # ------------------------------
        Q = self.W_q(x)                 # [B, L, D]
        K = self.W_k(extended_input)    # [B, L+M, D]
        V = self.W_v(extended_input)    # [B, L+M, D]

        # ------------------------------
        # Reshape for Multi-Head
        # ------------------------------
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)           # [B, n_heads, L, d_k]
        K = K.view(B, K.size(1), self.n_heads, self.head_dim).transpose(1, 2)   # [B, n_heads, L+M, d_k]
        V = V.view(B, V.size(1), self.n_heads, self.head_dim).transpose(1, 2)   # [B, n_heads, L+M, d_k]

        # ------------------------------
        # Scaled Dot-Product Attention
        # ------------------------------
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5) # [B, n_heads, L, L+M]
        attn_scores = torch.nan_to_num(attn_scores, nan=-1e9, posinf=1e4, neginf=-1e4)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

        context = torch.matmul(attn_probs, V)

        # ------------------------------
        # Multi-Head Merge
        # ------------------------------
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.out_proj(context) # [B, L, D]

        return output


