# main.py

import torch
import polars as pl
import sys

from resources.domain.data_loader_usecase import DataLoaderUseCase
from resources.service.model_runner_service import ModelRunnerService
from resources.service.preprocess_service import PreprocessService

# import mlflow
# mlflow.set_tracking_uri("file:///app/mlruns")
# mlflow.set_experiment("TitanForecasting")
# DIR = '/app/data'
# FILE_PATH = os.path.join(DIR, 'target_dyn_demand.parquet')


# 경로 설정
MAC_DIR = '../data/'
WINDOW_DIR = '/modeling_module/data/'

if sys.platform == 'win32':
    FILE_PATH = WINDOW_DIR
    print("[CUDA] Available:", torch.cuda.is_available())
    print("[CUDA] Device Count:", torch.cuda.device_count())
    print("[CUDA] Version:", torch.version.cuda)
    print("[CUDA] PyTorch Version:", torch.__version__)
    print("[CUDA] Device Name:", torch.cuda.get_device_name(0))
else:
    FILE_PATH = MAC_DIR


target_dyn_demand = pl.read_parquet(FILE_PATH)
def main():
    PreprocessService().run()

    train, val, inference = DataLoaderUseCase().run()

    # 2) 서비스 준비(로더 보관)
    svc = ModelRunnerService(train, val, inference)

if __name__ == "__main__":
    main()
