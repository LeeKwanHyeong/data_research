# Attention.py
import inspect
from math import sqrt
from typing import Dict, Type

import numpy as np
import torch
import torch.nn as nn

from modeling_module.utils.masking import TriangularCausalMask, ProbMask

# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------
ATTN_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_attention(name: str):
    def deco(cls):
        ATTN_REGISTRY[name] = cls
        return cls
    return deco


# ---------------------------------------------------------------------
# Full Attention (outputs [B, H, L, Dv])
# ---------------------------------------------------------------------
@register_attention('full')
class FullAttention(nn.Module):
    """
    Standard Scaled Dot-Product Attention (Multi-Head 지원)

    입력:
    - queries: [B, H, L, Dk]
    - keys:    [B, H, S, Dk]
    - values:  [B, H, S, Dv]
    - attn_mask: Causal mask 등 [B, H, L, S] 형태의 마스크 텐서

    출력:
    - out: [B, H, L, Dv] (attention 적용 결과)
    - attn(optional): [B, H, L, S] (softmax 결과)
    - logits(optional): [B, H, L, S] (raw dot-product 점수)

    지원 기능:
    - scale: Dk^-0.5 자동 scaling 또는 사용자 지정
    - mask_flag: causal attention 여부
    - output_attention: attention weight 반환 여부
    - RealFormer residual: 이전 logits에 residual 적용
    """
    def __init__(self,
                 mask_flag: bool = True,
                 factor: int = 5,                 # kept for interface parity
                 scale: float | None = None,
                 attention_dropout: float = 0.1,
                 output_attention: bool = False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # 기본값(안전)
        self.return_logits = False              # logits 반환 여부
        self.use_realformer_residual = False    # RealFormer residual 사용 여부

    def forward(self,
                queries: torch.Tensor,  # [B,H,L,Dk]
                keys: torch.Tensor,     # [B,H,S,Dk]
                values: torch.Tensor,   # [B,H,S,Dv]
                attn_mask: torch.Tensor | None,
                prev_logits: torch.Tensor | None = None):
        """
        Scaled Dot-Product Attention 연산 수행
        """
        B, H, L, Dk = queries.shape
        _, _, S, _ = keys.shape

        # --- Scaling ---
        scale = self.scale or 1.0 / sqrt(Dk)

        # --- Dot Product: [B, H, L, S] ---
        scores = torch.einsum('bhld,bhsd->bhls', queries, keys) * scale

        # --- RealFormer-style residual ---
        if self.use_realformer_residual and (prev_logits is not None):
            # prev_logits expected: [B,H,L,S]
            scores = scores + prev_logits

        # --- Causal Masking ---
        if self.mask_flag:
            if attn_mask is None:
                # Causal mask 자동 생성 (Triangular 형태)
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attn_mask.mask, neg_inf)

        # --- Numerical stability (clipping) ---
        scores = torch.clamp(scores, min=-50, max=50)

        # --- Softmax → Dropout ---
        A = torch.softmax(scores, dim=-1)    # [B,H,L,S]
        A = self.dropout(A)

        # --- Weighted sum: [B, H, L, Dv] ---
        V = torch.einsum('bhls,bhsd->bhld', A, values)

        # --- 반환 옵션 처리 ---
        if self.output_attention and self.return_logits:
            return V.contiguous(), A, scores
        elif self.output_attention:
            return V.contiguous(), A
        elif self.return_logits:
            return V.contiguous(), None, scores
        else:
            return V.contiguous(), None


# ---------------------------------------------------------------------
# Full Attention (logits 반환/잔차 가능)
# ---------------------------------------------------------------------
@register_attention('fullwithlogits')
class FullAttentionWithLogits(FullAttention):
    """
    FullAttention 확장 버전
    - logits 반환(return_logits=True)
    - RealFormer residual 적용(use_realformer_residual=True)

    기존 FullAttention과의 차이점:
    기본값으로 logits 출력 가능
    이전 logits 잔차(prev_logits)를 자동 처리

    등록 이름: 'fullwithlogits'
    """
    def __init__(self,
                 mask_flag: bool = True,
                 factor: int = 5,                       # 사용되지 않음 (인터페이스 구색 맞추기)
                 scale: float | None = None,
                 attention_dropout: float = 0.1,
                 output_attention: bool = False,
                 return_logits: bool = True,            # logits 반환 기본 활성화
                 use_realformer_residual: bool = True): # RealFormer residual 기본 활성화
        super().__init__(
            mask_flag=mask_flag,
            factor=factor,
            scale=scale,
            attention_dropout=attention_dropout,
            output_attention=output_attention,
        )
        self.return_logits = return_logits
        self.use_realformer_residual = use_realformer_residual

    def forward(self, queries, keys, values, attn_mask, prev_logits=None):
        """
        입력:
        - queries: [B, H, L, Dk]
        - keys:    [B, H, S, Dk]
        - values:  [B, H, S, Dv]
        - attn_mask: [B, H, L, S] or TriangularCausalMask 객체
        - prev_logits: [B, H, L, S] or None

        출력:
        - out: [B, H, L, Dv]
        - attention_weights: [B, H, L, S] (optional)
        - logits: [B, H, L, S] (optional)
        """
        return super().forward(queries, keys, values, attn_mask, prev_logits)


# ---------------------------------------------------------------------
# ProbSparse Attention (outputs [B, H, L, Dv])
#  - 입력/내부/출력 전부 [B,H,L,*] 축 규약으로 통일
#  - 복잡도O(LlogL)
#  - Linformer, Informer 등에서 사용
# ---------------------------------------------------------------------
@register_attention('probsparse')
class ProbAttention(nn.Module):
    def __init__(self,
                 mask_flag: bool = True,
                 factor: int = 5,
                 scale: float | None = None,
                 attention_dropout: float = 0.1,
                 output_attention: bool = False):
        """
        Args:
            mask_flag (bool): causal mask 활성화 여부
            factor (int): 샘플링 및 topk 크기 조절 계수
            scale (float): attention score scaling (보통 1/√d)
            attention_dropout (float): softmax 이후 dropout
            output_attention (bool): attention weights 반환 여부
        """
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    # -------------------------------------------------------------
    # top-k query 위치 찾기 위한 확률적 QK 샘플링
    # -------------------------------------------------------------
    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor, sample_k: int, n_top: int):
        """
        Args:
            Q, K: [B, H, L_Q/L_K, D]
            sample_k: K에서 샘플링할 개수 (U_part)
            n_top: Top Q 위치 수 (u)

        Returns:
            Q_K: Top-k Q에 대한 K와의 attention score [B, H, u, L_K]
            M_top: Top-k Q 인덱스 [B, H, u]
        """
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # K에서 무작위로 sample_k개를 Q별로 샘플링
        K_expand = K.unsqueeze(2).expand(B, H, L_Q, L_K, D)     # [B, H, L_Q, L_K, D]
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)  # [L_Q, sample_k]
        K_sample = K_expand[:, :, torch.arange(L_Q, device=Q.device).unsqueeze(1), index_sample, :]  # [B, H, L_Q, sample_k, D]

        # Q와 샘플링된 K의 내적 → sparse 측정
        Q_K_sample = torch.matmul(Q.unsqueeze(3), K_sample.transpose(-2, -1)).squeeze(3)  # [B, H, L_Q, sample_k]

        # sparsity measure: max - avg
        M = Q_K_sample.max(-1).values - (Q_K_sample.sum(-1) / L_K)  # [B, H, L_Q]

        # Top-n_top Q 인덱스 선택
        M_top = M.topk(n_top, dim=-1, sorted=False).indices         # [B, H, n_top]

        # 선택된 Top Q만 사용하여 K 전체와 attention score 계산
        Q_reduce = Q.gather(dim=2, index=M_top.unsqueeze(-1).expand(-1, -1, -1, D))  # [B, H, n_top, D]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # [B, H, n_top, L_K]
        return Q_K, M_top  # scores(top), indices


    # -------------------------------------------------------------
    # Initial Context 생성 (Full 평균 or 누적합)
    # -------------------------------------------------------------
    def _get_initial_context(self, V: torch.Tensor, L_Q: int):
        """
        Returns initial context:
        - 비마스킹: 평균 벡터를 전체 L_Q 위치에 복사
        - 마스킹: 누적합 (causal attention)
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_mean = V.mean(dim=2)  # [B, H, D]
            context = V_mean.unsqueeze(2).expand(B, H, L_Q, D).clone()  # [B,H,L_Q,D]
        else:
            assert L_Q == L_V, "Causal self-attn requires L_Q==L_V"
            context = V.cumsum(dim=2)  # [B,H,L_Q,D]
        return context

    # -------------------------------------------------------------
    # Top Query 위치에 대한 context 갱신
    # -------------------------------------------------------------
    def _update_context(self, context_in: torch.Tensor, V: torch.Tensor,
                        scores_top: torch.Tensor, index: torch.Tensor,
                        L_Q: int):
        """
        Args:
            context_in: 초기 context [B, H, L_Q, D]
            scores_top: [B, H, n_top, L_V]
            index: top-k query indices [B, H, n_top]
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            # ProbSparse용 마스크 적용
            pmask = ProbMask(B, H, L_Q, index, scores_top, device=V.device)
            scores_top = scores_top.masked_fill(pmask.mask, -np.inf)

        attn = torch.softmax(scores_top, dim=-1)  # [B, H, n_top, L_V]
        ctx_update = torch.matmul(attn, V)  # [B, H, n_top, D]

        # top-k 위치에만 context 업데이트
        context = context_in.clone()  # [B, H, L_Q, D]
        index_exp = index.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, H, n_top, D]
        context.scatter_(dim=2, index=index_exp, src=ctx_update) # top 위치만 update

        if self.output_attention:
            # Dense attn map for visualization (costly): [B,H,L_Q,L_V]
            # 전체 attention map 반환 (희소 주의: 비용 큼)
            attns = (torch.ones([B, H, L_Q, L_V], device=V.device, dtype=attn.dtype) / L_V)
            attns.scatter_(dim=2, index=index.unsqueeze(-1).expand(-1, -1, -1, L_V), src=attn)
            return context, attns
        else:
            return context, None

    # -------------------------------------------------------------
    # Forward: Q, K, V 입력 → ProbSparse Attention 수행
    # -------------------------------------------------------------
    def forward(self,
                queries: torch.Tensor,  # [B,H,L,Dk]
                keys: torch.Tensor,     # [B,H,S,Dk]
                values: torch.Tensor,   # [B,H,S,Dv]
                ):

        B, H, L_Q, Dk = queries.shape
        _, _, L_K, Dv = values.shape

        # 샘플링 파라미터 (log 기반으로 scale)
        U_part = min(self.factor * int(np.ceil(np.log(L_K))), L_K)  # sample size
        u = min(self.factor * int(np.ceil(np.log(L_Q))), L_Q)       # top-k queries

        # Top-k Q 선택 및 점수 계산
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)  # [B,H,u,L_K], [B,H,u]

        # scale 적용
        scale = self.scale or 1.0 / sqrt(Dk)
        scores_top = scores_top * scale

        # 초기 context 및 context 업데이트
        context = self._get_initial_context(values, L_Q)  # [B,H,L_Q,Dv]
        context, attn = self._update_context(context, values, scores_top, index, L_Q)  # [B,H,L_Q,Dv], opt attn

        return context.contiguous(), attn   # [B, H, L, Dv], optional attn


# ---------------------------------------------------------------------
# Thin Attention Wrappers
#  - Attention 모듈을 감싸 projection과 head 분할 처리 담당
# --------------
class AttentionLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int, d_keys=None, d_values=None):
        """
        Args:
            attention (nn.Module): FullAttention, ProbAttention 등 내부 attention 모듈
            d_model (int): 전체 hidden dimension
            n_heads (int): multi-head attention의 head 수
            d_keys (int): 각 head의 key/query 차원 (기본값: d_model // n_heads)
            d_values (int): 각 head의 value 차원 (기본값: d_model // n_heads)
        """
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values

    def forward(self, queries, keys, values, attn_mask):
        """
        Args:
            queries, keys, values: [B, L, d_model] 또는 [B, S, d_model]
            attn_mask: optional mask

        Returns:
            output: [B, L, d_model]
            attn (optional): attention weights
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Linear projection 후, multi-head 분할 → [B, H, L/S, D]
        Q = self.query_projection(queries).view(B, L, H, self.d_keys).transpose(1, 2)  # [B,H,L,Dk]
        K = self.key_projection(keys).view(B, S, H, self.d_keys).transpose(1, 2)       # [B,H,S,Dk]
        V = self.value_projection(values).view(B, S, H, self.d_values).transpose(1, 2) # [B,H,S,Dv]

        # Inner attention 적용 → [B,H,L,Dv], attn weights
        out_b_h_l_d, attn = self.inner_attention(Q, K, V, attn_mask)  # [B,H,L,Dv]

        # 다시 head 차원 합치기 → [B,L,H*Dv]
        out = out_b_h_l_d.transpose(1, 2).reshape(B, L, H * self.d_values)            # [B,L,H*Dv]

        # 최종 output projection → [B,L,d_model]
        return self.out_projection(out), attn


# ---------------------------------------------------------------------
# Wrapper for Attention with Residual Logits (RealFormer style)
# ---------------------------------------------------------------------
class AttentionLayerWithPrev(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int, d_keys=None, d_values=None):
        """
        FullAttentionWithLogits 등 residual이 필요한 attention을 위한 wrapper
        """
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values

    def forward(self, queries, keys, values, attn_mask, prev_logits=None):
        """
        Args:
            prev_logits: [B,H,L,S] residual logits (RealFormer)
        Returns:
            output: [B, L, d_model]
            attn (optional): attention weights
            logits (optional): raw attention logits
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        Q = self.query_projection(queries).view(B, L, H, self.d_keys).transpose(1, 2)
        K = self.key_projection(keys).view(B, S, H, self.d_keys).transpose(1, 2)
        V = self.value_projection(values).view(B, S, H, self.d_values).transpose(1, 2)

        # FullAttentionWithLogits 등에서 logits도 반환할 수 있음
        out_b_h_l_d, attn, *maybe_scores = self.inner_attention(Q, K, V, attn_mask, prev_logits=prev_logits)
        out = out_b_h_l_d.transpose(1, 2).reshape(B, L, H * self.d_values)
        out = self.out_projection(out)

        # logits 반환 여부에 따라 다르게 처리
        if len(maybe_scores) == 1:
            return out, attn, maybe_scores[0]  # logits
        return out, attn


# ---------------------------------------------------------------------
# MultiHeadAttention Wrapper
#  - 입력: [B, L, D] (Query, Key, Value 모두)
#  - 출력: [B, L, D] + (attention weights, logits)
#  - 내부 attention core는 반드시 [B, H, L, Dv] 반환해야 함
# ---------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 n_heads: int,
                 d_k: int | None = None,
                 d_v: int | None = None,
                 proj_dropout: float = 0.0,
                 attn_core: nn.Module,
                 **_):
        """
        Args:
            d_model (int): 입력 및 출력 벡터 차원
            n_heads (int): multi-head 개수
            d_k (int, optional): key/query의 head별 차원 (default: d_model // n_heads)
            d_v (int, optional): value의 head별 차원 (default: d_model // n_heads)
            proj_dropout (float): 최종 출력에 적용할 dropout 비율
            attn_core (nn.Module): FullAttention, ProbAttention 등 Attention core 모듈
        """
        super().__init__()
        d_k = (d_model // n_heads) if d_k is None else d_k
        d_v = (d_model // n_heads) if d_v is None else d_v

        # Linear projections: Q, K, V → [B, L/S, H * d_k or d_v]
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=True)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=True)
        self.W_V = nn.Linear(d_model, n_heads * d_v, bias=True)

        # Core attention module (ex: FullAttention, ProbAttention)
        self.core = attn_core
        self.core_accepts_prev = 'prev_logits' in inspect.signature(self.core.forward).parameters

        # Output projection: [B, L, H*Dv] → [B, L, D]
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model),
            nn.Dropout(proj_dropout),
        )

    def forward(self, Q, K=None, V=None, attn_mask=None, prev_logits=None):
        """
        Args:
            Q, K, V: [B, L, D]
            attn_mask: optional attention mask
            prev_logits: (optional) RealFormer-style residual logits

        Returns:
            output: [B, L, D]
            attn: [B, H, L, S] or None
            logits: [B, H, L, S] or None
        """
        if K is None: K = Q
        if V is None: V = Q
        B, L, _ = Q.shape
        S = K.size(1)

        # Project Q, K, V → [B, H, L/S, Dk or Dv]
        q = self.W_Q(Q).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # [B,H,L,Dk]
        k = self.W_K(K).view(B, S, self.n_heads, self.d_k).transpose(1, 2)  # [B,H,S,Dk]
        v = self.W_V(V).view(B, S, self.n_heads, self.d_v).transpose(1, 2)  # [B,H,S,Dv]

        # Call attention core
        if self.core_accepts_prev:
            core_out = self.core(q, k, v, attn_mask, prev_logits=prev_logits)
        else:
            core_out = self.core(q, k, v, attn_mask)

        # 결과 unpack
        if isinstance(core_out, tuple):
            if len(core_out) == 3:
                out_b_h_l_d, attn, logits = core_out
            else:
                out_b_h_l_d, attn = core_out
                logits = None
        else:
            out_b_h_l_d, attn, logits = core_out, None, None

        # 체크: core에서 4D 텐서 반환 필요
        assert out_b_h_l_d.dim() == 4, f"attn core must return 4D [B,H,L,Dv], got {out_b_h_l_d.shape}"

        # Merge multi-heads: [B, H, L, Dv] -> [B, L, H*Dv]
        B2, Hh, L_real, Dv = out_b_h_l_d.shape
        out = out_b_h_l_d.transpose(1, 2).contiguous().view(B2, L_real, Hh * Dv)  # [B,L,H*Dv]

        # Output projection (in_features auto-fix for safety)
        # Linear in_features 자동 보정 (체크포인트/설정 변경 시 안전)
        lin = self.to_out[0] if isinstance(self.to_out, nn.Sequential) else self.to_out
        if lin.in_features != out.shape[-1]:
            # in_features가 바뀌었을 경우 자동으로 to_out의 Linear 레이어 재설정
            new_lin = nn.Linear(out.shape[-1], lin.out_features, bias=(lin.bias is not None)).to(out.device)
            nn.init.xavier_uniform_(new_lin.weight)
            if new_lin.bias is not None:
                nn.init.zeros_(new_lin.bias)
            if isinstance(self.to_out, nn.Sequential):
                self.to_out[0] = new_lin
            else:
                self.to_out = new_lin

        # Apply final output projection
        out = self.to_out(out)  # [B, L, d_model]
        return out, attn, logits


# ---------------------------------------------------------------------
# Attention Builder: cfg를 기반으로 MultiHeadAttention 모듈 생성
# ---------------------------------------------------------------------
def build_attention(cfg) -> MultiHeadAttention:
    # Attention Core 선택
    core_name = cfg.type.lower()
    if core_name == 'full' and getattr(cfg, 'residual_logits', False):
        # FullAttention인데 residual logits 사용 시, 전용 core로 변경
        core_name = 'fullwithlogits'

    core_cls = ATTN_REGISTRY[core_name] # Attention core 클래스 선택

    # Core 클래스에 전달할 파라미터 구성
    core_kwargs = dict(
        mask_flag=getattr(cfg, 'causal', True), # Causal masking 여부
        attention_dropout=cfg.attn_dropout,     # dropout rate
        output_attention=getattr(cfg, 'output_attention', False), # attention map 반환 여부
    )
    if core_name == 'probsparse':
        core_kwargs['factor'] = cfg.factor  # ProbSparse 전용 하이퍼파라미터
    if core_name == 'fullwithlogits':
        core_kwargs['use_realformer_residual'] = True   # RealFormer 잔차 사용
        core_kwargs['return_logits'] = getattr(cfg, 'output_attention', False)
    if core_name == 'full':
        # residual logits가 full에서도 켜지도록 허용 (옵션)
        core_kwargs['use_realformer_residual'] = getattr(cfg, 'residual_logits', False)

    # Attention core 인스턴스 생성
    core = core_cls(**core_kwargs)

    # QK, V의 차원 수 설정
    d_k = cfg.d_k if cfg.d_k is not None else cfg.d_model // cfg.n_heads
    d_v = cfg.d_v if cfg.d_v is not None else cfg.d_model // cfg.n_heads

    # MultiHeadAttention 래퍼로 감싸서 반환
    return MultiHeadAttention(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_k=d_k,
        d_v=d_v,
        proj_dropout=cfg.proj_dropout,
        attn_core=core,
        # 아래 키워드는 MHA에서는 사용하지 않지만, 빌더 인터페이스 호환을 위해 받아만 둠
        attn_dropout=cfg.attn_dropout,
        causal=getattr(cfg, 'causal', True),
        residual_logits=getattr(cfg, 'residual_logits', False),
        output_attention=getattr(cfg, 'output_attention', False),
        factor=getattr(cfg, 'factor', 5),
        lsa=getattr(cfg, 'lsa', False),
    )
