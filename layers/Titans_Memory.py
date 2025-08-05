import torch
import torch.nn as nn
import torch.nn.functional as F

# Efficient Multi-Head Attention with Memory (Contextual + Persistent)
class MemoryAttention(nn.Module):
    def __init__(self, d_model, n_heads, contextual_mem_size, persistent_mem_size):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

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
        device = self.persistent_memory.device  # 모델 파라미터 기준 디바이스
        # new_context = new_context.to(device)
        new_context = new_context.detach()
        '''
            new_context: (B, M, d_model) - New contextual memory tokens
        '''
        if self.contextual_memory is None:
            self.contextual_memory = new_context

        else:
            # Append and keep only the last N tokens
            self.contextual_memory = torch.cat([self.contextual_memory, new_context], dim = 1)
            if self.contextual_memory.size(1) > self.contextual_mem_size:
                self.contextual_memory = self.contextual_memory[:, -self.contextual_mem_size:, :]


    def forward(self, x):
        '''
            x: (B, L, d_model) (Core input tokens)
        '''
        B, L, _ = x.size()

        # Build extended input: [contextual memory + persistent memory + input]
        persistent_mem_expanded = self.persistent_memory.unsqueeze(0).expand(B, -1, -1) # (B, P, d_model)
        contextual_mem = self.contextual_memory if self.contextual_memory is not None else torch.empty(B, 0, self.d_model, device = x.device)
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
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

# Transformer Block (Core)
class MemoryTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, contextual_mem_size, persistent_mem_size, dropout = 0.1):
        super().__init__()
        self.attn = MemoryAttention(d_model, n_heads, contextual_mem_size, persistent_mem_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
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
            MemoryTransformerBlock(d_model, n_heads, d_ff, contextual_mem_size, persistent_mem_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return x

