import math
from typing import Tuple

import torch.nn as nn
import torch
from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.PatchTST.common.heads.classification_head import ClassificationHead
from modeling_module.models.PatchTST.common.heads.prediction_head import PredictionHead
from modeling_module.models.PatchTST.common.heads.regression_head import RegressionHead
from modeling_module.models.PatchTST.self_supervised.backbone import SelfSupervisedBackbone
from modeling_module.models.PatchTST.supervised.backbone import SupervisedBackbone

"""
(a) Backbone (Supervised)
    Input: Univariate Series
    - Instance Norm(RevIN) + Patching
    - Projection + Position Embedding
    - Transformer Encoder
    - Flatten + Linear Head
    Output: Univariate Series
(b) Backbone (Self-supervised)
    Input: Univariate Series
    - Instance Norm(RevIN) + Patching
    - Projection + Position Embedding
    - Transformer Encoder
    - Linear Layer
    Output: Reconstructed Masked Patches
    
(a), (b) Transformer Encoder
    Inputs
    - Input Embedding
    - Multi-Head Attention
    - Add & Norm
    - Feed Forward
    - Add & Norm
"""

# ---------------------------------------
# 유틸: 패치 개수 계산 (lookback, patch_len, stride, padding)
# ---------------------------------------
def _num_patches(lookback: int, patch_len: int, stride: int, padding_patch: str | None) -> int:
    if padding_patch == "end":
        # 패딩으로 마지막 패치 하나를 더 확보
        return math.floor((lookback - 1) / stride) + 1
    # 일반 케이스: 정확히 끊어지지 않으면 마지막 일부는 버림
    return max(1, math.floor((lookback - patch_len) / stride) + 1)

# ---------------------------------------
# 유틸: head 타입/하이퍼 해석 (HeadConfig 우선, 폴백은 top-level)
# ---------------------------------------
def _resolve_head(cfg: PatchTSTConfig) -> Tuple[str, bool, float, tuple | None]:
    # type
    htype = getattr(cfg, "head", None).type if getattr(cfg, "head", None) else None
    if htype is None or htype == "flatten":
        # flatten은 프로젝트 내 구현 차가 크니 prediction/regression/classification 중 하나로 맵핑
        # top-level 레거시 값이 있으면 우선
        htype = getattr(cfg, "head_type", None) or "regression"
    # individual
    individual = getattr(cfg, "head", None).individual if getattr(cfg, "head", None) else getattr(cfg, "individual", False)
    # dropout
    head_dropout = getattr(cfg, "head", None).head_dropout if getattr(cfg, "head", None) else getattr(cfg, "head_dropout", 0.0)
    # y_range (회귀 한정)
    y_range = getattr(cfg, "head", None).y_range if getattr(cfg, "head", None) else None
    return htype, bool(individual), float(head_dropout), y_range

class PatchTSTModel(nn.Module):
    """
    출력 규약:
      - prediction : [bs, target_dim, nvars]
      - regression : [bs, target_dim]
      - classification : [bs, target_dim]
      - pretrain  : [bs, num_patch, nvars, patch_len]
    입력 규약(백본 구현에 따름):
      - 보통 (B, L, C) 원시시계열을 받아 내부에서 패칭합니다.
      - 현재 프로젝트 백본 입력이 [bs, num_patch, patch_len]이라면 그 형식을 그대로 사용하세요.
    """
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.cfg = config

        # --------- 파생 하이퍼 계산 ----------
        self.n_vars = self.cfg.c_in
        self.num_patch = _num_patches(self.cfg.lookback, self.cfg.patch_len, self.cfg.stride, self.cfg.padding_patch)

        # Head 해석(HeadConfig 우선)
        self.head_type, self.individual, self.head_dropout, self.y_range = _resolve_head(self.cfg)

        # --------- Backbone 선택 ----------
        if self.head_type == "pretrain":
            # 자기지도(마스킹 재구성) 백본
            self.backbone = SelfSupervisedBackbone(self.cfg)
        else:
            # 감독 학습 백본
            self.backbone = SupervisedBackbone(self.cfg)

        # --------- Head 구성 ----------
        if self.head_type == "pretrain":
            from modeling_module.models.PatchTST.common.heads.pretrain_head import PretrainHead
            self.head = PretrainHead(
                self.cfg.d_model,
                self.cfg.patch_len,
                self.cfg.dropout
            )
        elif self.head_type == "prediction":
            # 다변량 예측: [bs, target_dim, nvars]
            self.head = PredictionHead(
                self.individual,
                self.n_vars,
                self.cfg.d_model,
                self.num_patch,
                self.cfg.target_dim,
                self.head_dropout
            )
        elif self.head_type == "regression":
            # 전역 회귀: [bs, target_dim]
            self.head = RegressionHead(
                self.n_vars,
                self.cfg.d_model,
                self.cfg.target_dim,
                self.head_dropout,
                self.y_range
            )
        elif self.head_type == "classification":
            # 분류: [bs, target_dim]
            self.head = ClassificationHead(
                self.n_vars,
                self.cfg.d_model,
                self.cfg.target_dim,
                self.head_dropout
            )
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

        # (선택) Decomposition 훅을 백본 밖에서 처리하고 싶다면 여기서 구성
        # if self.cfg.decomp and self.cfg.decomp.type != "none":
        #     from models.components.decomposition import CausalMovingAverage1D
        #     self.decomp = CausalMovingAverage1D(window=self.cfg.decomp.ma_window)
        #     self.decomp_mode = self.cfg.decomp.concat_mode  # 'concat'|'residual_only'|'trend_only'
        # else:
        #     self.decomp, self.decomp_mode = None, None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: 백본이 기대하는 입력 텐서 형식으로 전달.
           - SupervisedBackbone이 (B,L,C)를 받도록 구현되어 있다면 그 형식으로 넘기고,
           - SelfSupervisedBackbone이 (B, num_patch, patch_len)를 받도록 구현되어 있으면 그 형식으로 넘기세요.
        """
        # (선택) 분해 훅이 외부라면 여기서 적용
        # if self.decomp is not None and z.dim() == 3 and z.shape[1] == self.cfg.lookback:
        #     # 예: (B,L,C) → (B,C,L) 변환 후 분해/concat/맵핑 등
        #     ...

        feats = self.backbone(z)    # 백본 출력 형식은 각 헤드가 기대하는 토큰/표현
        out = self.head(feats)
        return out





