from training.adapters import PatchMixerAdapter
from training.config import TrainingConfig
from training.engine import CommonTrainer


def train_patchmixer(model, train_loader, val_loader, **overrides):
    cfg = TrainingConfig(
        loss_mode="auto",              # (B,Q,H)이면 자동 quantile
        point_loss="mse",
        quantiles=(0.1,0.5,0.9),
        use_intermittent=True,
        val_use_weights=False,
        **overrides
    )
    trainer = CommonTrainer(cfg, PatchMixerAdapter())
    best_model = trainer.fit(model, train_loader, val_loader, tta_steps=0)
    return best_model