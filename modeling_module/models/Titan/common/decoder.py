import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Causal Decoder Layer ---
"""
   Input: tgt [B, H, D], memory [B, L, D]
        ↓
   ① norm1 + causal self-attention
        ↓
   ② residual connection + dropout
        ↓
   ③ norm2 + cross-attention with encoder
        ↓
   ④ residual connection + dropout
        ↓
   ⑤ norm3 + FFN
        ↓
   ⑥ final residual connection
        ↓
   Output: [B, H, D]
"""
class TitanDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout = 0.1, causal = True):
        """
        Titan 모델용 Causal Transformer Decoder Layer

        Parameters:
        - d_model: 임베딩 차원
        - n_heads: 멀티헤드 어텐션 헤드 수
        - d_ff: FeedForward 차원
        - dropout: 드롭아웃 확률
        - causal: causal masking 적용 여부 (True일 경우 미래 정보 차단)
        """
        super().__init__()

        # --- Causal Self-Attention ---
        self.self_attn = nn.MultiheadAttention(
            embed_dim = d_model, num_heads = n_heads, dropout = dropout, batch_first = True
        )

        # --- Cross-Attention (Encoder-Decoder Attention) ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim = d_model, num_heads = n_heads, dropout = dropout, batch_first = True
        )

        # --- Feed Forward Network (FFN) ---
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # --- LayerNorms ---
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.causal = causal

    def _causal_mask(self, H, device):
        """
        Causal mask 생성: 상삼각 마스크로 미래 정보 차단
        Shape: [H, H] (각 시점에 대해 그 이후 시점 차단)
        """
        # 상삼각(미래 차단)
        m = torch.full((H, H), float("-inf"), device = device)
        return torch.triu(m, diagonal = 1)

    def forward(self, tgt, memory):
        """
        Forward 함수

        Parameters:
        - tgt: 디코더 입력 (예: 이전 예측 값들), Shape: [B, H, D]
        - memory: 인코더 출력 (context vector), Shape: [B, L, D]

        Returns:
        - out: 디코더 출력, Shape: [B, H, D]
        """
        # tgt: [B, H, D], memory: [B, L, D]
        # --- Causal Self-Attention ---
        q = self.norm1(tgt)
        attn_mask = self._causal_mask(q.size(1), q.device) if self.causal else None
        x, _ = self.self_attn(q, q, q, attn_mask = attn_mask)   # masked self-attn
        x = tgt + self.dropout(x)   # Residual connection

        # --- Cross Attention ---
        y = self.norm2(x)
        z, _ = self.cross_attn(y, memory, memory)   # Encoder-Decoder attention
        x = x + self.dropout(z) # Residual connection

        # --- Feed Forward Network (FFN) ---
        w = self.norm3(x)
        f = self.ffn(w) # [B, H, D]
        out = x + f     # Final residual
        return out

# --- Causal Decoder Stack ---
class TitanDecoder(nn.Module):
    """
    Titan 모델의 디코더 스택 (Causal Transformer 기반)

    Parameters:
    - d_model: 임베딩 차원
    - n_layers: 디코더 레이어 수
    - n_heads: 멀티헤드 어텐션 헤드 수
    - d_ff: FFN 차원
    - dropout: 드롭아웃 확률
    - horizon: 예측 시계열 길이
    - exo_dim: 미래 외생변수 차원 (없으면 0)
    - causal: causal masking 적용 여부
    """
    def __init__(self,
                 d_model: int,
                 n_layers: int = 1,
                 n_heads: int = 4,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 horizon: int = 60,
                 exo_dim: int = 0,
                 causal: bool = True):
        super().__init__()
        self.horizon = horizon
        self.exo_dim = exo_dim

        # --- decoder 입력 쿼리 (query embedding) ---
        self.query_embed = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        # --- position embedding ---
        self.pos_embed   = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        # --- 외생변수 투영 (있을 경우) ---
        self.exo_proj = nn.Linear(exo_dim, d_model) if exo_dim > 0 else None

        # --- TitanDecoderLayer Stack 구성 ---
        self.layers = nn.ModuleList([
            TitanDecoderLayer(d_model, n_heads, d_ff, dropout, causal)
            for _ in range(n_layers)
        ])

    def forward(self,
                memory: torch.Tensor,                 # 인코더 출력 [B, L, D]
                future_exo: torch.Tensor | None=None  # 미래 외생변수 [B, H, exo_dim] or None
                ):
        """
        Forward 함수

        Parameters:
        - memory: 인코더 출력 (과거 context), shape = [B, L, D]
        - future_exo: 미래 외생변수, shape = [B, H, exo_dim] (없을 수도 있음)

        Returns:
        - x: 디코더 출력, shape = [B, H, D]
        """
        B, L, D = memory.shape
        H = self.horizon

        # --- 쿼리 임베딩 + 위치 임베딩 ---
        # (B, H, D) 형태로 브로드 캐스트
        tgt = self.query_embed.expand(B, H, D) + self.pos_embed.expand(B, H, D)

        # --- 미래 외생변수가 있을 경우 projection 후 합산 ---
        if (self.exo_proj is not None) and (future_exo is not None):
            # future_exo: [B, H, exo_dim]
            tgt = tgt + self.exo_proj(future_exo)

        # --- Decoder Layer 순차 적용 ---
        x = tgt
        for layer in self.layers:
            x = layer(x, memory)   # input: x [B, H, D], memory [B, L, D]

        return x    # 디코더 출력: [B, H, D]

