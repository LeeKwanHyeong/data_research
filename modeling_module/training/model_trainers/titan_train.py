from modeling_module.models.Titan.Titans import TestTimeMemoryManager
from modeling_module.training.adapters import TitanAdapter
from modeling_module.training.config import TrainingConfig
from modeling_module.training.engine import CommonTrainer
from modeling_module.utils.exogenous_utils import calendar_cb


def train_titan(model, train_loader, val_loader, *, tta_steps=0, **overrides):

    base = dict(
        loss_mode="point", point_loss="huber",
    )
    base.update(**overrides)
    cfg = TrainingConfig(**base)

    def factory(m): return TestTimeMemoryManager(m, lr=cfg.lr)  # 기존 로직
    adapter = TitanAdapter(tta_manager_factory=factory)

    trainer = CommonTrainer(cfg, adapter, future_exo_cb = calendar_cb)
    best_model = trainer.fit(model, train_loader, val_loader, tta_steps=tta_steps)
    return best_model