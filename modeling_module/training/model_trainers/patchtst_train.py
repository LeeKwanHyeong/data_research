from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.config import TrainingConfig
from modeling_module.training.engine import CommonTrainer
from modeling_module.utils.exogenous_utils import calendar_cb


def train_patchtst(model, train_loader, val_loader, **overrides):
    base = dict(
        loss_mode="auto",  # 출력 차원에 따라 자동
        point_loss="mse",
        quantiles=(0.1, 0.5, 0.9),
        use_intermittent=True,
        val_use_weights=False,
    )
    base.update(overrides)
    cfg = TrainingConfig(**base)

    # PatchTST는 보통 forward 단항, 별도 reg 없음 → DefaultAdapter
    trainer = CommonTrainer(cfg, DefaultAdapter(), future_exo_cb = calendar_cb)
    best_model = trainer.fit(model, train_loader, val_loader)
    return best_model