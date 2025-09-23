from training.adapters import DefaultAdapter
from training.config import TrainingConfig
from training.engine import CommonTrainer


def train_patchtst(model, train_loader, val_loader, **overrides):
    cfg = TrainingConfig(
        loss_mode="auto",           # 출력 차원에 따라 자동
        point_loss="mse",
        quantiles=(0.1,0.5,0.9),
        use_intermittent=True,
        val_use_weights=False,
        **overrides
    )
    # PatchTST는 보통 forward 단항, 별도 reg 없음 → DefaultAdapter
    trainer = CommonTrainer(cfg, DefaultAdapter())
    best_model = trainer.fit(model, train_loader, val_loader)
    return best_model