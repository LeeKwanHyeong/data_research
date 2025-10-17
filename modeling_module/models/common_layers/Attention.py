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
        self.return_logits = False
        self.use_realformer_residual = False

    def forward(self,
                queries: torch.Tensor,  # [B,H,L,Dk]
                keys: torch.Tensor,     # [B,H,S,Dk]
                values: torch.Tensor,   # [B,H,S,Dv]
                attn_mask: torch.Tensor | None,
                prev_logits: torch.Tensor | None = None):
        B, H, L, Dk = queries.shape
        _, _, S, _ = keys.shape

        scale = self.scale or 1.0 / sqrt(Dk)
        # [B,H,L,S]
        scores = torch.einsum('bhld,bhsd->bhls', queries, keys) * scale

        # RealFormer-style residual on logits
        if self.use_realformer_residual and (prev_logits is not None):
            # prev_logits expected: [B,H,L,S]
            scores = scores + prev_logits

        # causal mask
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attn_mask.mask, neg_inf)

        scores = torch.clamp(scores, min=-50, max=50)
        A = torch.softmax(scores, dim=-1)    # [B,H,L,S]
        A = self.dropout(A)

        # === 핵심: out은 반드시 [B,H,L,Dv] ===
        V = torch.einsum('bhls,bhsd->bhld', A, values)

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
    def __init__(self,
                 mask_flag: bool = True,
                 factor: int = 5,
                 scale: float | None = None,
                 attention_dropout: float = 0.1,
                 output_attention: bool = False,
                 return_logits: bool = True,
                 use_realformer_residual: bool = True):
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
        # 부모 클래스 로직 그대로 사용 (이미 [B,H,L,Dv] 반환)
        return super().forward(queries, keys, values, attn_mask, prev_logits)


# ---------------------------------------------------------------------
# ProbSparse Attention (outputs [B, H, L, Dv])
#  - 입력/내부/출력 전부 [B,H,L,*] 축 규약으로 통일
# ---------------------------------------------------------------------
@register_attention('probsparse')
class ProbAttention(nn.Module):
    def __init__(self,
                 mask_flag: bool = True,
                 factor: int = 5,
                 scale: float | None = None,
                 attention_dropout: float = 0.1,
                 output_attention: bool = False):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor, sample_k: int, n_top: int):
        # Q,K: [B,H,L,D]
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # sample K per query index
        # K_expand: [B,H,L_Q,L_K,D]
        K_expand = K.unsqueeze(2).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)  # U_part
        K_sample = K_expand[:, :, torch.arange(L_Q, device=Q.device).unsqueeze(1), index_sample, :]  # [B,H,L_Q,sample_k,D]
        # Q.unsqueeze(3): [B,H,L_Q,1,D]
        Q_K_sample = torch.matmul(Q.unsqueeze(3), K_sample.transpose(-2, -1)).squeeze(3)  # [B,H,L_Q,sample_k]

        # sparsity measurement
        M = Q_K_sample.max(-1).values - (Q_K_sample.sum(-1) / L_K)  # [B,H,L_Q]
        M_top = M.topk(n_top, dim=-1, sorted=False).indices         # [B,H,n_top]

        # reduce Q for top queries
        # gather along L_Q
        # M_top expansion: [B,H,n_top,D]
        Q_reduce = Q.gather(dim=2, index=M_top.unsqueeze(-1).expand(-1, -1, -1, D))  # [B,H,n_top,D]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # [B,H,n_top,L_K]
        return Q_K, M_top  # scores(top), indices

    def _get_initial_context(self, V: torch.Tensor, L_Q: int):
        # V: [B,H,L_V,D]
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_mean = V.mean(dim=2)  # [B,H,D]
            context = V_mean.unsqueeze(2).expand(B, H, L_Q, D).clone()  # [B,H,L_Q,D]
        else:
            assert L_Q == L_V, "Causal self-attn requires L_Q==L_V"
            context = V.cumsum(dim=2)  # [B,H,L_Q,D]
        return context

    def _update_context(self, context_in: torch.Tensor, V: torch.Tensor,
                        scores_top: torch.Tensor, index: torch.Tensor,
                        L_Q: int, attn_mask):
        # V: [B,H,L_V,D], scores_top: [B,H,n_top,L_V], index: [B,H,n_top]
        B, H, L_V, D = V.shape

        if self.mask_flag:
            pmask = ProbMask(B, H, L_Q, index, scores_top, device=V.device)
            scores_top = scores_top.masked_fill(pmask.mask, -np.inf)

        attn = torch.softmax(scores_top, dim=-1)  # [B,H,n_top,L_V]
        # matmul: [B,H,n_top,L_V] x [B,H,L_V,D] -> [B,H,n_top,D]
        ctx_update = torch.matmul(attn, V)  # [B,H,n_top,D]

        # scatter update into corresponding top positions
        context = context_in.clone()  # [B,H,L_Q,D]
        # index: [B,H,n_top] -> expand to D
        index_exp = index.unsqueeze(-1).expand(-1, -1, -1, D)  # [B,H,n_top,D]
        context.scatter_(dim=2, index=index_exp, src=ctx_update)

        if self.output_attention:
            # Dense attn map for visualization (costly): [B,H,L_Q,L_V]
            attns = (torch.ones([B, H, L_Q, L_V], device=V.device, dtype=attn.dtype) / L_V)
            attns.scatter_(dim=2, index=index.unsqueeze(-1).expand(-1, -1, -1, L_V), src=attn)
            return context, attns
        else:
            return context, None

    def forward(self,
                queries: torch.Tensor,  # [B,H,L,Dk]
                keys: torch.Tensor,     # [B,H,S,Dk]
                values: torch.Tensor,   # [B,H,S,Dv]
                attn_mask):
        B, H, L_Q, Dk = queries.shape
        _, _, L_K, Dv = values.shape

        U_part = min(self.factor * int(np.ceil(np.log(L_K))), L_K)  # sample size
        u = min(self.factor * int(np.ceil(np.log(L_Q))), L_Q)       # top-k queries

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)  # [B,H,u,L_K], [B,H,u]

        scale = self.scale or 1.0 / sqrt(Dk)
        scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)  # [B,H,L_Q,Dv]
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)  # [B,H,L_Q,Dv], opt attn

        return context.contiguous(), attn


# ---------------------------------------------------------------------
# Thin wrappers (optional; kept for compatibility)
# ---------------------------------------------------------------------
class AttentionLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int, d_keys=None, d_values=None):
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
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        Q = self.query_projection(queries).view(B, L, H, self.d_keys).transpose(1, 2)  # [B,H,L,Dk]
        K = self.key_projection(keys).view(B, S, H, self.d_keys).transpose(1, 2)       # [B,H,S,Dk]
        V = self.value_projection(values).view(B, S, H, self.d_values).transpose(1, 2) # [B,H,S,Dv]

        out_b_h_l_d, attn = self.inner_attention(Q, K, V, attn_mask)  # [B,H,L,Dv]
        out = out_b_h_l_d.transpose(1, 2).reshape(B, L, H * self.d_values)            # [B,L,H*Dv]
        return self.out_projection(out), attn


class AttentionLayerWithPrev(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int, d_keys=None, d_values=None):
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
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        Q = self.query_projection(queries).view(B, L, H, self.d_keys).transpose(1, 2)
        K = self.key_projection(keys).view(B, S, H, self.d_keys).transpose(1, 2)
        V = self.value_projection(values).view(B, S, H, self.d_values).transpose(1, 2)

        out_b_h_l_d, attn, *maybe_scores = self.inner_attention(Q, K, V, attn_mask, prev_logits=prev_logits)
        out = out_b_h_l_d.transpose(1, 2).reshape(B, L, H * self.d_values)
        out = self.out_projection(out)

        if len(maybe_scores) == 1:
            return out, attn, maybe_scores[0]  # logits
        return out, attn


# ---------------------------------------------------------------------
# MHA wrapper: input (B,L,D) -> output (B,L,D)
#   - assumes core returns [B,H,L,Dv]
#   - auto-fixes to_out Linear in_features if H*Dv changed
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
        super().__init__()
        d_k = (d_model // n_heads) if d_k is None else d_k
        d_v = (d_model // n_heads) if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=True)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=True)
        self.W_V = nn.Linear(d_model, n_heads * d_v, bias=True)

        self.core = attn_core
        self.core_accepts_prev = 'prev_logits' in inspect.signature(self.core.forward).parameters

        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model),
            nn.Dropout(proj_dropout),
        )

    def forward(self, Q, K=None, V=None, attn_mask=None, prev_logits=None):
        if K is None: K = Q
        if V is None: V = Q
        B, L, _ = Q.shape
        S = K.size(1)

        q = self.W_Q(Q).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # [B,H,L,Dk]
        k = self.W_K(K).view(B, S, self.n_heads, self.d_k).transpose(1, 2)  # [B,H,S,Dk]
        v = self.W_V(V).view(B, S, self.n_heads, self.d_v).transpose(1, 2)  # [B,H,S,Dv]

        if self.core_accepts_prev:
            core_out = self.core(q, k, v, attn_mask, prev_logits=prev_logits)
        else:
            core_out = self.core(q, k, v, attn_mask)

        if isinstance(core_out, tuple):
            if len(core_out) == 3:
                out_b_h_l_d, attn, logits = core_out
            else:
                out_b_h_l_d, attn = core_out
                logits = None
        else:
            out_b_h_l_d, attn, logits = core_out, None, None

        assert out_b_h_l_d.dim() == 4, f"attn core must return 4D [B,H,L,Dv], got {out_b_h_l_d.shape}"
        B2, Hh, L_real, Dv = out_b_h_l_d.shape

        out = out_b_h_l_d.transpose(1, 2).contiguous().view(B2, L_real, Hh * Dv)  # [B,L,H*Dv]

        # Linear in_features 자동 보정 (체크포인트/설정 변경 시 안전)
        lin = self.to_out[0] if isinstance(self.to_out, nn.Sequential) else self.to_out
        if lin.in_features != out.shape[-1]:
            new_lin = nn.Linear(out.shape[-1], lin.out_features, bias=(lin.bias is not None)).to(out.device)
            nn.init.xavier_uniform_(new_lin.weight)
            if new_lin.bias is not None:
                nn.init.zeros_(new_lin.bias)
            if isinstance(self.to_out, nn.Sequential):
                self.to_out[0] = new_lin
            else:
                self.to_out = new_lin

        out = self.to_out(out)  # [B, L, d_model]
        return out, attn, logits


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------
def build_attention(cfg) -> MultiHeadAttention:
    # 1) choose core
    core_name = cfg.type.lower()
    if core_name == 'full' and getattr(cfg, 'residual_logits', False):
        core_name = 'fullwithlogits'

    core_cls = ATTN_REGISTRY[core_name]

    # 2) core kwargs
    core_kwargs = dict(
        mask_flag=getattr(cfg, 'causal', True),
        attention_dropout=cfg.attn_dropout,
        output_attention=getattr(cfg, 'output_attention', False),
    )
    if core_name == 'probsparse':
        core_kwargs['factor'] = cfg.factor
    if core_name == 'fullwithlogits':
        core_kwargs['use_realformer_residual'] = True
        core_kwargs['return_logits'] = getattr(cfg, 'output_attention', False)
    if core_name == 'full':
        # residual logits가 full에서도 켜지도록 허용 (옵션)
        core_kwargs['use_realformer_residual'] = getattr(cfg, 'residual_logits', False)

    core = core_cls(**core_kwargs)

    # 3) d_k/d_v
    d_k = cfg.d_k if cfg.d_k is not None else cfg.d_model // cfg.n_heads
    d_v = cfg.d_v if cfg.d_v is not None else cfg.d_model // cfg.n_heads

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
