import polars as pl

from modeling_module.data_loader.TimeSeriesModule import MultiPartDataModule
from modeling_module.training.config import TrainingConfig
from resources.engine_config import IS_RUNNING, DIR

class DataLoaderUseCase:
    def __init__(self):
        if IS_RUNNING:
            self.target_df = pl.read_parquet(DIR + 'target_dyn_demand_weekly.parquet')
        else:
            self.target_df = pl.read_parquet(DIR + 'target_dyn_demand_monthly.parquet')


    def run(self):
        plan_dt = 202305

        train_cfg = TrainingConfig()

        data_module = MultiPartDataModule(
            self.target_df,
            train_cfg,
            batch_size=64,
            val_ratio=0.2,
            is_running=IS_RUNNING
        )
        train_loader = data_module.get_train_loader()
        val_loader = data_module.get_val_loader()
        inference_loader = data_module.get_inference_loader_at_plan(plan_dt)

        return train_loader, val_loader, inference_loader