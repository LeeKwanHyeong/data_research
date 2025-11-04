from typing import Optional, Callable
from modeling_module.training.adapters import PatchMixerAdapter
from modeling_module.training.config import TrainingConfig
from modeling_module.training.engine import CommonTrainer

def train_patchmixer(
    model,
    train_loader,
    val_loader,
    *,
    future_exo_cb: Optional[Callable] = None,   # ← 외생변수 생성 콜백
    exo_is_normalized: bool = True,             # (필요 시 어댑터/모델에서 사용할 힌트)
    **overrides
):
    """
    TrainingConfig를 건드리지 않고 외생변수를 사용하려면:
      - 외생 콜백(future_exo_cb)을 CommonTrainer에 넘겨주기만 하면 됨.
      - TrainingConfig에는 exo 관련 필드가 없어도 무방.
    """
    # TrainingConfig에는 존재하는 키만 넣어야 하므로, config 관련 키만 추립니다.
    base_cfg = dict(
        loss_mode="auto",
        point_loss="mae",
        quantiles=(0.1, 0.5, 0.9),
        use_intermittent=True,
        val_use_weights=False,
    )
    # overrides 중 TrainingConfig에 없는 키 제거 (ex: future_exo_cb, exo_is_normalized 등)
    cfg_keys = set(base_cfg.keys()) | {
        "device","lookback","horizon","epochs","lr","weight_decay","t_max",
        "patience","max_grad_norm","amp_device","huber_delta","q_star",
        "use_cost_q_star","Cu","Co","quantiles","use_intermittent","alpha_zero",
        "alpha_pos","gamma_run","cap","use_horizon_decay","tau_h","val_use_weights"
    }
    clean_overrides = {k:v for k,v in overrides.items() if k in cfg_keys}
    base_cfg.update(clean_overrides)
    cfg = TrainingConfig(**base_cfg)

    # 스파이크 친화 손실(옵션)이 필요하면 cfg.spike_loss를 여기서 조정 가능
    # 예) 혼합전략
    # cfg.spike_loss.enabled = True
    # cfg.spike_loss.strategy = 'mix'
    # 예) 단일형(Huber 비대칭)
    # cfg.point_loss = 'huber_asym'
    # cfg.spike_loss.enabled = True
    # cfg.spike_loss.strategy = 'direct'

    # 어댑터가 exo_is_normalized 플래그를 쓸 수 있게 전달(필요 없다면 제거해도 됨)
    # PatchMixerAdapter 가 kwargs를 받아서 forward로 넘긴다면 활용, 아니라면 무시됨.
    adapter = PatchMixerAdapter()  # 안전: 사용 안 하면 영향 없음

    trainer = CommonTrainer(
        cfg,
        adapter,
        logger=print,
        metrics_fn=None,
        future_exo_cb=future_exo_cb,  # ← 핵심: 콜백만 전달하면 트레이너가 (B,H,D) 만들어 모델로 넘깁니다.
    )
    best_model = trainer.fit(model, train_loader, val_loader, tta_steps=0)
    print(f"[EXO-train] model.exo_dim={getattr(model, 'exo_dim', 0)}  "
          f"exo_head? {hasattr(model, 'exo_head') and (model.exo_head is not None)}  "
          f"future_exo_cb? {future_exo_cb is not None}")

    return best_model
