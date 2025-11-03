from modeling_module.training.adapters import PatchMixerAdapter
from modeling_module.training.config import TrainingConfig
from modeling_module.training.engine import CommonTrainer, TrainerCallbacks
from modeling_module.utils.exogenous_utils import calendar_cb


def train_patchmixer(model, train_loader, val_loader, **overrides):
    base = dict(
        loss_mode="auto",
        point_loss="mae",
        quantiles=(0.1, 0.5, 0.9),
        use_intermittent=True,
        val_use_weights=False,
    )
    base.update(overrides)  # 호출부 인자로 덮어쓰기
    cfg = TrainingConfig(**base)

    trainer = CommonTrainer(cfg, PatchMixerAdapter(),
                            logger=print,
                            # metrics_fn=my_metrics_fn,        # 선택
                            future_exo_cb=calendar_cb,         # 선택
                            callbacks=TrainerCallbacks(      # 선택
                                    on_epoch_end=lambda ep, info: print("ep end:", ep, info)
                                ),
                            )
    best_model = trainer.fit(model, train_loader, val_loader, tta_steps=0)
    return best_model