from resources.domain.data_loader_usecase import DataLoaderUseCase
import polars as pl
import sys

from resources.engine_config import IS_RUNNING, DIR


class PreprocessService:
    def __init__(self):

        self.data_loader_uc = DataLoaderUseCase()

    def run(self):
        raw_target_df = pl.read_parquet(DIR + 'target_dyn_demand.parquet')

        if IS_RUNNING:
            target_df = self._make_target_table_weekly(df = raw_target_df)
        else:
            target_df = self._make_target_table_monthly(df = raw_target_df)

        return target_df



    def _make_target_table_weekly(self, df: pl.DataFrame):
        out = pl.read_parquet(DIR + 'target_dyn_demand_weekly.parquet')
        return out

    def _make_target_table_monthly(self, df: pl.DataFrame):
        out = pl.read_parquet(DIR + 'target_dyn_demand_monthly.parquet')
        return out