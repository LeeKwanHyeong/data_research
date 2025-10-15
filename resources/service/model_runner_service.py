# model_runner_service.py
from __future__ import annotations
from typing import Iterable, Optional, Dict, Any, Callable
import polars as pl

from modeling_module.utils.exogenous_utils import calendar_cb
from resources.domain.forecast_usecase import ForecastUseCase
from resources.domain.training_usecase import TrainingUseCase


class ModelRunnerService:
    def __init__(
        self,
        train_loader: Optional[Iterable] = None,
        val_loader: Optional[Iterable] = None,
        inference_loader: Optional[Iterable] = None,
        *,
        logger: Callable[[str], None] = print,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.inference_loader = inference_loader
        self._logger = logger

    # ---------------- Forecast ----------------
    def run_forecast(
        self,
        *,
        data_loader: Optional[Iterable] = None,
        device: str = "cuda",
        future_exo_cb: Optional[Callable] = calendar_cb,
        horizon: int = 120,
        target_channel: int = 0,
    ) -> pl.DataFrame:
        data_loader = data_loader or self.inference_loader
        if data_loader is None:
            raise ValueError("inference_loader가 설정되지 않았고 data_loader도 전달되지 않았습니다.")

        uc = ForecastUseCase(
            device=device,
            future_exo_cb=future_exo_cb,
            horizon=horizon,
            target_channel=target_channel,
            logger=self._logger,
        )
        return uc.execute(data_loader=data_loader)  # req dict 아님

    # ---------------- Training ----------------
    def run_training(
        self,
        *,
        train_loader: Optional[Iterable] = None,
        val_loader: Optional[Iterable] = None,
        device: str = "cuda",
        return_summary: bool = True,
    ) -> Dict[str, Any]:
        train_loader = train_loader or self.train_loader
        val_loader = val_loader or self.val_loader
        if train_loader is None or val_loader is None:
            raise ValueError("train_loader 또는 val_loader가 없습니다. 서비스 생성자에서 설정하거나 인자로 전달해 주세요.")

        uc = TrainingUseCase(device=device, logger=self._logger)
        return uc.execute(train_loader=train_loader, val_loader=val_loader, return_summary=return_summary)
