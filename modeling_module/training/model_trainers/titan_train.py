from typing import Optional, Callable

from modeling_module.models.Titan.Titans import TestTimeMemoryManager
from modeling_module.training.adapters import TitanAdapter
from modeling_module.training.config import TrainingConfig
from modeling_module.training.engine import CommonTrainer
from modeling_module.utils.exogenous_utils import calendar_sin_cos
import torch

def train_titan(
    model,
    train_loader,
    val_loader,
    *,
    future_exo_cb: Optional[Callable] = None,  # ← 외생변수 생성 콜백(선택)
    exo_is_normalized: bool = True,            # ← 어댑터/모델에서 필요시 사용할 힌트
    tta_steps: int = 0,
    **overrides
):
    """
    Titan 학습 트레이너(외생변수 지원, 침습 최소화).
    - future_exo_cb가 주어지면 해당 콜백을 사용해 (B,H,D_exo) 생성 후 모델에 전달.
    - future_exo_cb가 None이면 다음 우선순위로 자동 선택:
        1) 모델이 exo_dim > 0 이고, 모델(또는 내부 config)에 use_calendar_exo가 True → calendar_sin_cos 사용
        2) 그 외 → 외생변수 미사용(None)
    - TrainingConfig에 존재하지 않는 override 키는 정리하여 안전 적용.
    """

    # ===== 1) TrainingConfig 기본값 + overrides 정리 =====
    base_cfg = dict(
        loss_mode="point",      # Titan 기본
        point_loss="huber",
    )

    # TrainingConfig에서 허용하는 키들만 통과(패턴: patchmixer_train.py와 유사)
    cfg_keys = {
        "device", "lookback", "horizon", "epochs", "lr", "weight_decay",
        "t_max", "patience", "max_grad_norm", "amp_device",
        "loss_mode", "point_loss", "huber_delta",
        "quantiles", "q_star", "use_cost_q_star", "Cu", "Co",
        "use_intermittent", "alpha_zero", "alpha_pos", "gamma_run", "cap",
        "use_horizon_decay", "tau_h", "val_use_weights"
    }
    clean_overrides = {k: v for k, v in overrides.items() if k in cfg_keys}
    base_cfg.update(clean_overrides)
    cfg = TrainingConfig(**base_cfg)

    # ===== 2) 외생변수 콜백 선택 로직 =====
    # - 가장 명시적인 사용자의 인자(future_exo_cb)를 우선.
    # - 없으면 모델 설정을 점검해 calendar_sin_cos를 기본 제공.
    #   (configs.py: use_calendar_exo / exo_dim 참조)
    #   모델이 config 속성을 보관하지 않아도 exo_dim 기준으로 결정 가능.
    if future_exo_cb is not None:
        fe_cb = future_exo_cb
    else:
        use_calendar = False
        # 모델이 내부에 config를 보관하는 경우 우대
        model_cfg = getattr(model, "config", None)
        if model_cfg is not None:
            use_calendar = bool(getattr(model_cfg, "use_calendar_exo", False))
            exo_dim = int(getattr(model_cfg, "exo_dim", getattr(model, "exo_dim", 0)))
        else:
            exo_dim = int(getattr(model, "exo_dim", 0))

        if not use_calendar:
            # config가 없거나 False여도, exo_dim>0이면 calendar를 기본 제공
            use_calendar = (exo_dim > 0)

        fe_cb = calendar_sin_cos if use_calendar else None

    # ===== 3) TTA 매니저/어댑터 구성(기존 로직 유지) =====
    def factory(m):  # 기존 Test-Time Memory Manager 유지
        return TestTimeMemoryManager(m, lr=cfg.lr)

    adapter = TitanAdapter(
        tta_manager_factory=factory,
        # 필요 시 어댑터가 참조하도록 힌트 전달(사용하지 않으면 무시)
    )

    amp_device: str = "cuda"
    amp_dtype: str = "bf16"  # "fp16" 대신 기본 "bf16" 권장 (5080 지원)
    amp_enabled: bool = True
    use_bf16 = (amp_dtype.lower() == "bf16")
    autocast_input = dict(
        device_type=amp_device,
        enabled=amp_enabled,
        dtype=(torch.bfloat16 if use_bf16 else torch.float16),
    )

    # ===== 4) CommonTrainer 실행 =====
    trainer = CommonTrainer(
        cfg,
        adapter,
        logger=print,
        metrics_fn=None,
        future_exo_cb=fe_cb,    # ← 핵심: 외생변수 콜백 전달
        autocast_input = autocast_input,
    )
    best_model = trainer.fit(model, train_loader, val_loader, tta_steps=tta_steps)

    # ===== 5) 로깅 =====
    print(
        f"[EXO-train] exo_dim={getattr(model, 'exo_dim', 0)} "
        f"exo_head? {hasattr(model, 'exo_head') and (getattr(model, 'exo_head') is not None)} "
        f"future_exo_cb? {fe_cb is not None} "
        f"tta_steps={tta_steps}"
    )

    return best_model
