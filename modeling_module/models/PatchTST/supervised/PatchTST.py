
import torch
from torch import nn
from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.PatchTST.supervised.backbone import SupervisedBackbone
from modeling_module.models.common_layers.RevIN import RevIN


class PointHeadWithExo(nn.Module):
    """
    Backbone 출력 + Future Exo -> Horizon 예측
    """

    def __init__(self, d_model: int, horizon: int, d_future: int, patch_num: int, agg: str = "mean"):
        super().__init__()
        self.agg = agg
        self.horizon = horizon
        self.d_future = d_future

        # Future Exo Projection: [B, H, d_future] -> [B, d_model]
        # 미래 정보를 d_model 크기로 압축하여 문맥에 더하거나 concat
        self.future_proj = nn.Linear(horizon * d_future, d_model) if d_future > 0 else None

        # Final Projection
        # Input: d_model (Backbone) + d_model (Future) if concat else d_model
        # 여기서는 단순하게 Backbone Feat + Future Feat (Concat) 전략 사용
        in_dim = d_model * 2 if d_future > 0 else d_model

        self.proj = nn.Linear(in_dim, horizon)

    def forward(self, z_bld: torch.Tensor, future_exo: torch.Tensor = None) -> torch.Tensor:
        if self.agg == "mean":
            feat = z_bld.mean(dim=1)
        else:
            feat = z_bld[:, -1, :]

        if self.d_future > 0:
            if future_exo is None:
                raise RuntimeError(
                    f"[PatchTST] d_future={self.d_future}인데 future_exo가 None입니다. "
                    f"Adapter/forward 시그니처 호환을 확인하세요."
                )
            B, H, D = future_exo.shape
            if D != self.d_future:
                raise RuntimeError(
                    f"[PatchTST] future_exo last-dim(D)={D} != d_future={self.d_future}"
                )

            f_flat = future_exo.reshape(B, -1)
            f_feat = self.future_proj(f_flat)
            feat = torch.cat([feat, f_feat], dim=-1)

        return self.proj(feat)

class QuantileHeadWithExo(nn.Module):
    """
    (B, L_tok, d_model) + Future Exo -> (B, H, Q)
    - Future Exo: [B, H, d_future]를 (H*d_future)로 펼쳐 d_model로 projection 후 concat
    - 최종 출력: (B, H, Q)  (Q는 마지막 축)
    """
    def __init__(
        self,
        d_model: int,
        horizon: int,
        d_future: int,
        quantiles=(0.1, 0.5, 0.9),
        hidden: int = 128,
        monotonic: bool = True,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.quantiles = tuple(quantiles)
        self.Q = len(self.quantiles)
        self.monotonic = bool(monotonic)
        self.d_future = int(d_future)

        self.future_proj = nn.Linear(self.horizon * self.d_future, d_model) if self.d_future > 0 else None
        in_dim = d_model * 2 if self.d_future > 0 else d_model

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.horizon * self.Q),
        )

    def forward(self, z_bld: torch.Tensor, future_exo: torch.Tensor = None) -> torch.Tensor:
        B = z_bld.size(0)
        feat = z_bld.mean(dim=1)  # [B, d_model]

        if self.d_future > 0:
            if future_exo is None:
                raise RuntimeError(
                    f"[PatchTST-Quantile] d_future={self.d_future}인데 future_exo가 None입니다."
                )
            if future_exo.dim() == 2:  # (H,E) -> (B,H,E)
                future_exo = future_exo.unsqueeze(0).expand(B, -1, -1)

            if future_exo.dim() != 3:
                raise RuntimeError(f"[PatchTST-Quantile] future_exo must be 3D, got {tuple(future_exo.shape)}")

            b2, H, D = future_exo.shape
            if b2 != B:
                raise RuntimeError(f"[PatchTST-Quantile] future_exo batch mismatch: {b2} != {B}")
            if H != self.horizon:
                # horizon이 다른 호출(IMS 확장 등)을 허용하려면 여기 정책을 바꾸셔야 합니다.
                raise RuntimeError(f"[PatchTST-Quantile] future_exo horizon mismatch: {H} != {self.horizon}")
            if D != self.d_future:
                raise RuntimeError(
                    f"[PatchTST-Quantile] future_exo last-dim(D)={D} != d_future={self.d_future}"
                )

            f_flat = future_exo.reshape(B, -1)      # [B, H*D]
            f_feat = self.future_proj(f_flat)       # [B, d_model]
            feat = torch.cat([feat, f_feat], dim=-1)  # [B, 2*d_model]

        out = self.net(feat).view(B, self.horizon, self.Q)  # [B, H, Q]

        if self.monotonic:
            out, _ = torch.sort(out, dim=-1)  # quantile 축(Q) monotonic
        return out

class PatchTSTPointModel(nn.Module):
    def __init__(self, cfg, attn_core=None):
        super().__init__()
        self.cfg = cfg

        # Backbone (Past Exo 처리 포함)
        self.backbone = SupervisedBackbone(cfg, attn_core)

        # Patch Number 계산 (Head 초기화용)
        from modeling_module.models.PatchTST.common.patching import compute_patch_num
        patch_num = compute_patch_num(cfg.lookback, cfg.patch_len, cfg.stride, cfg.padding_patch)

        # Head (Future Exo 처리 포함)
        self.head = PointHeadWithExo(
            d_model=cfg.d_model,
            horizon=cfg.horizon,
            d_future=getattr(cfg, 'd_future', 0),
            patch_num=patch_num,
            agg="mean"  # or "last"
        )

        self.is_quantile = False
        self.horizon = cfg.horizon
        self.model_name = "PatchTST ExoModel"

        # RevIN은 Target 변수에만 적용 (Exo는 DataModule 단계나 별도 Norm 필요)
        self.revin_layer = RevIN(num_features=cfg.c_in)

    @classmethod
    def from_config(cls, config: "PatchTSTConfig"):
        return cls(cfg=config)

    def forward(
        self,
        x: torch.Tensor,
        # ---- New (Trainer/DefaultAdapter compatible) ----
        future_exo: torch.Tensor | None = None,
        past_exo_cont: torch.Tensor | None = None,
        past_exo_cat: torch.Tensor | None = None,
        part_ids=None,
        mode: str | None = None,
        # ---- Legacy aliases (backward-compat) ----
        fe_cont: torch.Tensor | None = None,
        pe_cont: torch.Tensor | None = None,
        pe_cat: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Trainer/Adapter가 넘기는 인자명(future_exo, past_exo_cont, past_exo_cat)과
        기존 PatchTST 인자명(fe_cont, pe_cont, pe_cat)을 모두 수용합니다.
        """

        # 0) alias resolve
        fe = future_exo if future_exo is not None else fe_cont
        p_cont = past_exo_cont if past_exo_cont is not None else pe_cont
        p_cat = past_exo_cat if past_exo_cat is not None else pe_cat

        # 1) RevIN normalize (기존 로직 유지)
        use_revin = getattr(self.cfg, "use_revin", True)
        x_n = self.revin_layer(x, "norm") if use_revin else x

        # 2) Backbone Forward (Past Exo)
        z = self.backbone(x_n, p_cont=p_cont, p_cat=p_cat)  # [B, N, d_model]

        # 3) Head Forward (Future Exo)
        y_n = self.head(z, future_exo=fe)  # [B, H]

        if use_revin:
            y = self.revin_layer(y_n.unsqueeze(-1), "denorm").squeeze(-1)  # [B,H]
            return y

        return y_n

class PatchTSTQuantileModel(nn.Module):
    def __init__(self, cfg, attn_core=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = SupervisedBackbone(cfg, attn_core)

        self.head = QuantileHeadWithExo(
            d_model=cfg.d_model,
            horizon=cfg.horizon,
            d_future=getattr(cfg, "d_future", 0),
            quantiles=getattr(cfg, "quantiles", (0.1, 0.5, 0.9)),
            hidden=getattr(cfg, "q_hidden", 128),
            monotonic=getattr(cfg, "monotonic_quantiles", True),
        )

        self.is_quantile = True
        self.horizon = cfg.horizon
        self.model_name = "PatchTST QuantileModel"

        self.revin_layer = RevIN(num_features=cfg.c_in)

    @classmethod
    def from_config(cls, config: "PatchTSTConfig"):
        return cls(cfg=config)

    def forward(
        self,
        x: torch.Tensor,
        # ---- New (Trainer/DefaultAdapter compatible) ----
        future_exo: torch.Tensor | None = None,
        past_exo_cont: torch.Tensor | None = None,
        past_exo_cat: torch.Tensor | None = None,
        part_ids=None,
        mode: str | None = None,
        # ---- Legacy aliases (backward-compat) ----
        fe_cont: torch.Tensor | None = None,
        pe_cont: torch.Tensor | None = None,
        pe_cat: torch.Tensor | None = None,
        **kwargs,
    ):
        use_revin = getattr(self.cfg, "use_revin", True)
        # alias resolve (Point 모델과 동일)
        fe = future_exo if future_exo is not None else fe_cont
        p_cont = past_exo_cont if past_exo_cont is not None else pe_cont
        p_cat = past_exo_cat if past_exo_cat is not None else pe_cat

        # 1) normalize (target only)
        x_n = self.revin_layer(x, "norm") if use_revin else x  # [B, L, C]

        # 2) backbone (past exo 포함)
        z = self.backbone(x_n, p_cont=p_cont, p_cat=p_cat)  # [B, N, d_model]

        # 3) head (future exo 포함)  -> [B, H, Q]
        q_n = self.head(z, future_exo=fe)

        if use_revin:
            if q_n.dim() == 2:
                # [B,H] -> [B,H,1]로 만들어 denorm
                q_den = self.revin_layer(q_n.unsqueeze(-1), "denorm").squeeze(-1)  # [B,H]
                return {"q": q_den}

            elif q_n.dim() == 3:
                # [B,H,Q] -> flatten 후 denorm -> reshape
                B, H, Q = q_n.shape
                q_flat = q_n.reshape(B, H * Q, 1)  # [B, H*Q, 1]
                q_den = self.revin_layer(q_flat, "denorm").reshape(B, H, Q)
                return {"q": q_den}

            else:
                raise RuntimeError(f"[PatchTSTQuantile] unexpected q_n.dim={q_n.dim()} shape={tuple(q_n.shape)}")

        return {"q": q_n}