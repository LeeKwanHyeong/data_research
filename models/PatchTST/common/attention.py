import inspect

import torch
import torch.nn as nn
import numpy as np
from math import sqrt

from utils.masking import TriangularCausalMask, ProbMask
from typing import Dict, Type, Optional, Tuple

ATTN_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_attention(name: str):
    def deco(cls):
        ATTN_REGISTRY[name] = cls
        return cls
    return deco


@register_attention('full')
class FullAttention(nn.Module):
    def __init__(self, mask_flag = True, factor = 5, scale = None, attention_dropout = 0.1, output_attention = False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, prev_logits=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        scale = self.scale or 1. / sqrt(E)
        # (B, H, L, S)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys) * scale

        if self.use_realformer_residual and (prev_logits is not None):
            scores = scores + prev_logits  # (B,H,L,S)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attn_mask.mask, neg_inf)

        A = torch.softmax(scores, dim=-1)
        A = self.dropout(A)

        # ✅ out shape 은 (B, L, H, d_v) 이어야 함
        V = torch.einsum('bhls,bshd->blhd', A, values)

        if self.output_attention and self.return_logits:
            return V.contiguous(), A, scores
        elif self.output_attention:
            return V.contiguous(), A
        elif self.return_logits:
            return V.contiguous(), None, scores
        else:
            return V.contiguous(), None


@register_attention('fullwithlogits')
class FullAttentionWithLogits(FullAttention):
    """
    - return_logits = True: (out, attn, logits) 반환
    - use_realformer_residual = True: 이전 logits(prev_logits)을 현재 logits에 더함
    """
    def __init__(self,
                 mask_flag = True,
                 factor = 5,
                 scale = None,
                 attention_dropout = 0.1,
                 output_attention = False,
                 return_logits = False,
                 use_realformer_residual = False):
        super().__init__(
            mask_flag = mask_flag,
            factor = factor,
            scale = scale,
            attention_dropout = attention_dropout,
            output_attention = output_attention
        )
        self.return_logits = return_logits
        self.use_realformer_residual = use_realformer_residual

    def forward(self, queries, keys, values, attn_mask, prev_logits = None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        scale = self.scale or 1. / sqrt(E)
        # (B, H, L, S)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys) * scale

        # RealFormer-style residual on logits
        if self.use_realformer_residual and (prev_logits is not None):
            # prev_logits expected shape: (B, H, L, S)
            scores = scores + prev_logits

        # masking
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device = queries.device)
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attn_mask.mask, neg_inf)

        A = torch.softmax(scores, dim = -1)
        A = self.dropout(A)

        V = torch.einsum('bhls,bshd->bhls', A, values)

        if self.output_attention and self.return_logits:
            return V.contiguous(), A, scores
        elif self.output_attention:
            return V.contiguous(), A
        elif self.return_logits:
            return V.contiguous(), None, scores
        else:
            return V.contiguous(), None

@register_attention('probsparse')
class ProbAttention(nn.Module):
    def __init__(self, mask_flag = True, factor = 5, scale = None, attention_dropout = 0.1, output_attention = False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor * ln(L_k)) * L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduce Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim = 2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert (L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            context = V.cumsum(dim = 2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device = V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim = -1)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k = U_part, n_top = u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        return context.contiguous(), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys = None, d_values = None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)
        return self.out_projection(out), attn

class AttentionLayerWithPrev(nn.Module):
    def __init__(self,
                 attention,
                 d_model,
                 n_heads,
                 d_keys = None,
                 d_values = None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, prev_logits = None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        Q = self.query_projection(queries).reshape(B, L, H, -1)
        K = self.key_projection(keys).reshape(B, S, H, -1)
        V = self.value_projection(values).reshape(B, S, H, -1)

        # 내부 커널이 prev_logits를 받도록 설계
        out, attn, *maybe_scores = self.inner_attention(Q, K, V, attn_mask, prev_logits = prev_logits)
        out = out.reshape(B, L, -1)
        out = self.out_projection(out)

        if len(maybe_scores) == 1:
            return out, attn, maybe_scores[0] # logits
        return out, attn

class MultiHeadAttention(nn.Module):
    """
    MHA 래퍼: 입력 (B,L,D) -> 출력 (B,L,D)
    - 코어(attn_core)는 (B,L,H,Dh) 입출력 컨벤션
    - 여분의 키워드 인자(**kwargs)는 받아서 무시 (호출 측 유연성 확보)
    """
    def __init__(self, *,
                 d_model: int,
                 n_heads: int,
                 d_k: int | None = None,
                 d_v: int | None = None,
                 proj_dropout: float = 0.0,
                 attn_core: nn.Module,
                 **_):  # ← 여분 키워드는 무시 (attn_dropout/causal/... 등)
        super().__init__()
        d_k = (d_model // n_heads) if d_k is None else d_k
        d_v = (d_model // n_heads) if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=True)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=True)
        self.W_V = nn.Linear(d_model, n_heads * d_v, bias=True)

        self.core = attn_core
        # 코어가 prev_logits를 받는지 자동 감지
        self.core_accepts_prev = 'prev_logits' in inspect.signature(self.core.forward).parameters

        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model),
            nn.Dropout(proj_dropout)
        )

    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor | None = None,
                V: torch.Tensor | None = None,
                attn_mask: torch.Tensor | None = None,
                prev_logits: torch.Tensor | None = None):
        # Q,K,V: (B,L,D)
        B, L, _ = Q.shape
        if K is None: K = Q
        if V is None: V = Q
        S = K.size(1)

        # (B,L,H,Dh)
        q = self.W_Q(Q).reshape(B, L, self.n_heads, self.d_k)
        k = self.W_K(K).reshape(B, S, self.n_heads, self.d_k)
        v = self.W_V(V).reshape(B, S, self.n_heads, self.d_v)

        # 코어 호출 (prev_logits 지원 여부에 따라 분기)
        if self.core_accepts_prev:
            core_out = self.core(q, k, v, attn_mask, prev_logits=prev_logits)
        else:
            core_out = self.core(q, k, v, attn_mask)

        # 코어 출력 정규화: (out, attn) 또는 (out, attn, logits) 모두 허용
        if isinstance(core_out, tuple):
            if len(core_out) == 3:
                out_b_l_h_d, attn, logits = core_out
            else:
                out_b_l_h_d, attn = core_out
                logits = None
        else:
            out_b_l_h_d, attn, logits = core_out, None, None

        out = out_b_l_h_d.reshape(B, L, self.n_heads * self.d_v)
        out = self.to_out(out)
        return out, attn, logits


def build_attention(cfg) -> MultiHeadAttention:
    # 1) core 선택 로직
    core_name = cfg.type
    if cfg.type == 'full' and getattr(cfg, 'residual_logits', False):
        core_name = 'fullwithlogits'

    core_cls = ATTN_REGISTRY[core_name]

    # 2) core kwargs (허용된 키만)
    core_kwargs = dict(
        mask_flag = getattr(cfg, 'causal', True),
        attention_dropout = cfg.attn_dropout,
        output_attention = getattr(cfg, 'output_attention', False),
    )
    if core_name == 'probsparse':
        core_kwargs['factor'] = cfg.factor
    if core_name == 'fullwithlogits':
        # 로짓 잔차 활성화
        core_kwargs['use_realformer_residual'] = True
        # 필요 시 로짓 반환 (분석/시각화용)
        core_kwargs['return_logits'] = getattr(cfg, 'output_attention', False)

    core = core_cls(**core_kwargs)

    # 3) d_k/d_v 기본값 계산
    d_k = cfg.d_k if cfg.d_k is not None else cfg.d_model // cfg.n_heads
    d_v = cfg.d_v if cfg.d_v is not None else cfg.d_model // cfg.n_heads

    return MultiHeadAttention(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_k=d_k,
        d_v=d_v,
        attn_dropout=cfg.attn_dropout,
        proj_dropout=cfg.proj_dropout,
        causal=getattr(cfg, 'causal', True),
        residual_logits=getattr(cfg, 'residual_logits', False),  # 래퍼 레벨에서 보관만(현재 미사용)
        output_attention=getattr(cfg, 'output_attention', False),
        factor=getattr(cfg, 'factor', 5),
        lsa=getattr(cfg, 'lsa', False),
        attn_core=core
    )