# training_usecase.py
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, Optional

from modeling_module.models.PatchMixer.common.configs import PatchMixerConfigMonthly
from modeling_module.models.PatchTST.common.configs import PatchTSTConfigMonthly
from modeling_module.models.Titan.common.configs import TitanConfigMonthly
from modeling_module.training.model_trainers.total_train import summarize_metrics, run_total_train_monthly
from modeling_module.utils.checkpoint import save_model_dict
from resources.engine_config import DIR


class TrainingUseCase:
    """
    total_train.run_total_train_monthly를 호출해 (여러 모델의) 학습을 수행하고,
    summarize_metrics로 요약 테이블을 만들어 반환.
    """
    def __init__(
        self,
        *,
        device: str = "cuda",
        logger: Callable[[str], None] = print,
        runner: Callable[[Iterable, Iterable, str], Dict[str, Dict]] = run_total_train_monthly,
        summarizer: Callable[[Dict[str, Dict]], Dict[str, Dict[str, float]]] = summarize_metrics,
    ):
        self.device = device
        self._logger = logger
        self._runner = runner
        self._summarizer = summarizer

        self.pm_config = PatchMixerConfigMonthly(
            device=self.device, loss_mode='quantile', quantiles=(0.1, 0.5, 0.9)
        )
        self.ti_config = TitanConfigMonthly(
            device=self.device, loss_mode='point', point_loss='huber'
        )
        self.pt_config = PatchTSTConfigMonthly(
            device=self.device, loss_mode='auto', quantiles=(0.1, 0.5, 0.9)
        )

    def execute(
        self,
        *,
        train_loader: Iterable,
        val_loader: Iterable,
        return_summary: bool = True,
    ) -> Dict[str, Any]:
        if train_loader is None or val_loader is None:
            raise ValueError("TrainingUseCase: train_loader 또는 val_loader가 없습니다.")

        self._logger("[TrainingUseCase] 학습 시작 (monthly)")
        results = self._runner(train_loader, val_loader, device=self.device)
        self._save_result(results)
        self._logger("[TrainingUseCase] 학습 종료")

        summary = self._summarizer(results) if return_summary else None
        if summary:
            self._logger(f"[TrainingUseCase] 요약 메트릭 keys: {list(summary.keys())[:5]}...")
        return {"results": results, "summary": summary}

    def _save_result(self, results: Dict[str, Any]) -> None:
        cfg_map = {
            "PatchMixer Base": self.pm_config,
            "PatchMixer Quantile": self.pm_config,
            "Titan Base": self.ti_config,
            "Titan LMM": self.ti_config,
            "Titan Seq2Seq": self.ti_config,
            "PatchTST Base": self.pt_config,
            "PatchTST Quantile": self.pt_config,
        }
        builder_key_by_name = {
            "PatchMixer Base": "patchmixer_base",
            "PatchMixer Quantile": "patchmixer_quantile",
            "Titan Base": "titan_base",
            "Titan LMM": "titan_lmm",
            "Titan Seq2Seq": "titan_seq2seq",
            "PatchTST Base": "patchtst_base",
            "PatchTST Quantile": "patchtst_quantile",
        }
        save_dir = DIR + 'fit'
        save_model_dict(results, save_dir, cfg_by_name=cfg_map, builder_key_by_name=builder_key_by_name)