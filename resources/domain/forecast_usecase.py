from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional, Any
import numpy as np
import polars as pl
import torch

from modeling_module.models.PatchMixer.common.configs import PatchMixerConfigMonthly
from modeling_module.models.PatchTST.common.configs import PatchTSTConfigMonthly
from modeling_module.models.Titan.common.configs import TitanConfigMonthly
from modeling_module.models.model_builder import (
    build_patch_mixer_base, build_patch_mixer_quantile,
    build_titan_base, build_titan_lmm, build_titan_seq2seq,
    build_patchTST_base, build_patchTST_quantile
)
from modeling_module.training.config import TrainingConfig
from modeling_module.utils.checkpoint import load_model_dict
from modeling_module.utils.plot_utils import _predict_120_any, _to_1d_history
from resources.engine_config import DIR

class ForecastUseCase:
    """
    __init__에서 전역 설정을 고정하고, execute()는 data_loader만 받아서
    체크포인트에서 로드된 self.loaded(모델 dict)를 사용해 예측 결과를 Polars DF로 반환.
    결과 스키마(롱 포맷): part_id | model | step | yhat | q10 | q50 | q90 | history_L
    """
    def __init__(
        self,
        *,
        device: str = "cuda",
        future_exo_cb: Optional[Callable] = None,
        horizon: int = 120,
        target_channel: int = 0,
        logger: Callable[[str], None] = print,
        predict_fn: Callable[..., Dict[str, Any]] = _predict_120_any,
        auto_load: bool = True,
    ):
        self._logger = logger
        self._predict = predict_fn

        self.device = device
        self.future_exo_cb = future_exo_cb
        self.horizon = int(horizon)
        self.target_channel = int(target_channel)

        self.builders = {
            "patchmixer_base": lambda cfg: build_patch_mixer_base(cfg or PatchMixerConfigMonthly()),
            "patchmixer_quantile": lambda cfg: build_patch_mixer_quantile(cfg or PatchMixerConfigMonthly()),
            "titan_base": lambda cfg: build_titan_base(cfg or TitanConfigMonthly()),
            "titan_lmm": lambda cfg: build_titan_lmm(cfg or TitanConfigMonthly()),
            "titan_seq2seq": lambda cfg: build_titan_seq2seq(cfg or TitanConfigMonthly()),
            "patchtst_base": lambda cfg: build_patchTST_base(cfg or PatchTSTConfigMonthly()),
            "patchtst_quantile": lambda cfg: build_patchTST_quantile(cfg or PatchTSTConfigMonthly()),
        }
        self.save_dir = DIR + 'fit'
        self.cfg = TrainingConfig()
        self.loaded: Dict[str, torch.nn.Module] = load_model_dict(self.save_dir, self.builders, device=self.device) if auto_load else {}

    @torch.no_grad()
    def execute(
        self,
        *,
        data_loader: Iterable,
        models: Optional[Dict[str, torch.nn.Module]] = None,  # 옵션: 외부 모델 dict로 self.loaded 대체 가능
    ) -> pl.DataFrame:
        models = models or self.loaded
        if not models:
            raise ValueError("ForecastUseCase: 사용할 모델이 없습니다. (self.loaded 비어 있음, models 인자도 없음)")

        rows: List[dict] = []
        H = self.horizon

        for batch_idx, batch in enumerate(data_loader):
            # (xb, yb, part_ids) 또는 (xb, yb)
            if len(batch) == 3:
                xb, yb, part_ids = batch
            else:
                xb, yb = batch
                part_ids = [f"idx{batch_idx:05d}_{i:04d}" for i in range(xb.size(0))]

            if xb.dim() == 2:
                xb = xb.unsqueeze(-1)  # (B,L) -> (B,L,1)

            B = xb.size(0)
            for i in range(B):
                x1 = xb[i:i+1].to(self.device)  # (1,L,C)
                # 필요 시 특정 채널만 사용하려면 다음 한 줄 활성화:
                # x1 = x1[..., [self.target_channel]]

                part_id = part_ids[i] if i < len(part_ids) else f"idx{batch_idx}_{i}"
                try:
                    hist_len = int(_to_1d_history(x1).shape[0])
                except Exception:
                    hist_len = int(x1.shape[1])

                for model_name, model in models.items():
                    pred = self._predict(model=model, x1=x1, device=self.device, future_exo_cb=self.future_exo_cb)

                    y_point = (np.asarray(pred.get("point")).reshape(-1)
                               if pred.get("point") is not None else None)

                    q10 = q50 = q90 = None
                    if isinstance(pred.get("q"), dict):
                        q10 = np.asarray(pred["q"].get("q10")).reshape(-1)
                        q50 = np.asarray(pred["q"].get("q50")).reshape(-1)
                        q90 = np.asarray(pred["q"].get("q90")).reshape(-1)

                    base = q50 if q50 is not None else y_point
                    if base is None:
                        raise RuntimeError(f"{model_name}: 예측이 비었습니다. 'point' 또는 'q50' 필요.")
                    if base.size != H:
                        raise RuntimeError(f"{model_name}: 예측 길이={base.size}, horizon={H} 불일치.")

                    if q50 is None:
                        for t in range(H):
                            rows.append({
                                "part_id": part_id, "model": model_name, "step": t+1,
                                "yhat": float(y_point[t]), "q10": None, "q50": None, "q90": None,
                                "history_L": hist_len,
                            })
                    else:
                        for t in range(H):
                            rows.append({
                                "part_id": part_id, "model": model_name, "step": t+1,
                                "yhat": float(q50[t]),
                                "q10": float(q10[t]), "q50": float(q50[t]), "q90": float(q90[t]),
                                "history_L": hist_len,
                            })

        df = pl.DataFrame(
            rows,
            schema={
                "part_id": pl.String, "model": pl.String, "step": pl.Int32,
                "yhat": pl.Float64, "q10": pl.Float64, "q50": pl.Float64, "q90": pl.Float64,
                "history_L": pl.Int32,
            }
        ).sort(["part_id", "model", "step"])

        self._logger(f"[ForecastUseCase] 결과 행수: {df.shape[0]} (parts × models × {H})")
        return df