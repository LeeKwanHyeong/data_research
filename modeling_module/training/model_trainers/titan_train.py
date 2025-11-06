from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Optional, Callable

import torch
from modeling_module.training.adapters import TitanAdapter, DefaultAdapter
from modeling_module.training.config import TrainingConfig
from modeling_module.training.engine import CommonTrainer
from modeling_module.utils.exogenous_utils import calendar_sin_cos


def _pick_future_exo_cb(model, user_cb: Optional[Callable]) -> Optional[Callable]:
    """
    외생변수 콜백 선택 우선순위:
    1) 사용자가 명시적으로 준 콜백
    2) model.config.use_calendar_exo=True 또는 exo_dim>0 → calendar_sin_cos
    3) 그 외 None
    """
    if user_cb is not None:
        return user_cb

    use_calendar = False
    exo_dim = int(getattr(model, "exo_dim", 0))

    model_cfg = getattr(model, "config", None)
    if model_cfg is not None:
        use_calendar = bool(getattr(model_cfg, "use_calendar_exo", False))
        exo_dim = int(getattr(model_cfg, "exo_dim", exo_dim))

    if not use_calendar:
        use_calendar = (exo_dim > 0)

    return calendar_sin_cos if use_calendar else None

def _dump_cfg(cfg):
    data = asdict(cfg) if is_dataclass(cfg) else cfg.__dict__
    print("[train_titan] Effective TrainingConfig:")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))

def train_titan(
    model,
    train_loader,
    val_loader,
    *,
    train_cfg: Optional[TrainingConfig] = None,
    future_exo_cb=None,
):
    """
    - train_cfg가 있으면 그 값을 우선 사용
    - 개별 인자(lr, loss_mode 등)는 train_cfg가 None일 때만 fallback으로 적용
    """

    adapter = DefaultAdapter()  # Titan 전용 어댑터 사용 중이면 여기서 교체

    _dump_cfg(train_cfg)

    # amp_device 기본값 처리
    amp_device = getattr(train_cfg, "amp_device", "cuda")

    # 사용자가 cfg에 amp_dtype를 문자열로 줄 수도 있으니 안전하게 해석
    amp_dtype_str = getattr(train_cfg, "amp_dtype", "bf16")
    if isinstance(amp_dtype_str, torch.dtype):
        amp_dtype = amp_dtype_str
    else:
        s = str(amp_dtype_str).lower()
        if s in ("bf16", "bfloat16"):
            amp_dtype = torch.bfloat16
        elif s in ("fp16", "float16", "half"):
            amp_dtype = torch.float16
        elif s in ("fp32", "float32"):
            amp_dtype = torch.float32
        else:
            amp_dtype = torch.bfloat16  # 기본값

    amp_enabled = (amp_device == "cuda" and torch.cuda.is_available())

    autocast_input = {
        "device_type": amp_device,
        "enabled": amp_enabled,
        "dtype": amp_dtype,  # ← 문자열이 아닌 torch.dtype 로 전달
    }

    trainer = CommonTrainer(
        cfg=train_cfg,
        adapter=adapter,
        future_exo_cb=future_exo_cb,
        logger=print,
        autocast_input=autocast_input,
    )
    model = trainer.fit(model, train_loader, val_loader, tta_steps=2)
    return {
        "model": model,
        "cfg": train_cfg,
    }

# def train_titan(
#     model,
#     train_loader,
#     val_loader,
#     *,
#     future_exo_cb: Optional[Callable] = None,  # 외생변수 생성 콜백(선택)
#     exo_is_normalized: bool = True,            # 어댑터/모델에서 필요 시 사용할 힌트
#     tta_steps: int = 0,
#     **overrides
# ):
#     """
#     Titan 학습 트레이너 (PatchMixer/PatchTST 스타일 정렬)
#     - TrainingConfig + CommonTrainer + TitanAdapter 사용
#     - model.config/use_calendar_exo/exo_dim 여부에 따라 캘린더 sin/cos 자동 주입
#     - TTA는 어댑터 훅만 유지(기본 비활성)
#     """
#     # 1) 기본 하이퍼파라미터 + 사용자 override
#     base_cfg = dict(
#         loss_mode="point",
#         point_loss="huber_asym",
#         use_intermittent=True,
#         val_use_weights=False,
#     )
#
#     # TrainingConfig 필드로 제한하여 깨끗하게 반영
#     allowed = {
#         "device","lookback","horizon","epochs","lr","weight_decay","t_max",
#         "patience","max_grad_norm","amp_device","huber_delta","q_star",
#         "use_cost_q_star","Cu","Co","quantiles","use_intermittent","alpha_zero",
#         "alpha_pos","gamma_run","cap","use_horizon_decay","tau_h","val_use_weights"
#     }
#     clean_overrides = {k: v for k, v in overrides.items() if k in allowed}
#     base_cfg.update(clean_overrides)
#     cfg = TrainingConfig(**base_cfg)
#
#     # 2) 외생변수 콜백 결정
#     fe_cb = _pick_future_exo_cb(model, future_exo_cb)
#
#     # 3) AMP 설정(bf16 추천)
#     amp_device: str = getattr(cfg, "amp_device", "cuda")
#     amp_enabled: bool = (amp_device == "cuda" and torch.cuda.is_available())
#     autocast_input = dict(
#         device_type=amp_device,
#         enabled=amp_enabled,
#         dtype=torch.bfloat16,
#     )
#
#     # 4) 공통 트레이너 실행
#     adapter = TitanAdapter()
#     trainer = CommonTrainer(
#         cfg,
#         adapter,
#         logger=print,
#         metrics_fn=None,
#         future_exo_cb=fe_cb,
#         autocast_input=autocast_input,
#     )
#
#     best_model = trainer.fit(model, train_loader, val_loader, tta_steps=tta_steps)
#
#     # 5) 로깅
#     print(
#         f"[EXO-train] exo_dim={getattr(model, 'exo_dim', 0)} "
#         f"exo_head? {hasattr(model, 'exo_head') and (getattr(model, 'exo_head') is not None)} "
#         f"future_exo_cb? {fe_cb is not None} "
#         f"tta_steps={tta_steps}"
#     )
#     return best_model
