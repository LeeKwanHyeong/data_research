from models.Titan.Titans import TestTimeMemoryManager
from training.adapters import TitanAdapter
from training.config import TrainingConfig
from training.engine import CommonTrainer


def train_titan(model, train_loader, val_loader, *, tta_steps=0, **overrides):
    cfg = TrainingConfig(loss_mode="point", point_loss="mse", **overrides)

    def factory(m): return TestTimeMemoryManager(m, lr=cfg.lr)  # 기존 로직
    adapter = TitanAdapter(tta_manager_factory=factory)

    trainer = CommonTrainer(cfg, adapter)
    best_model = trainer.fit(model, train_loader, val_loader, tta_steps=tta_steps)
    return best_model