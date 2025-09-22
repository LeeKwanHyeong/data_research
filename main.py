# main.py

import torch
import polars as pl
import sys

from data_loader.TimeSeriesModule import MultiPartDataModule
from model_runner.train.titanl_train import TitanTrain
from models.Titan.Titans import TitanConfigMonthly, LMMModel

# import mlflow
# mlflow.set_tracking_uri("file:///app/mlruns")
# mlflow.set_experiment("TitanForecasting")
# DIR = '/app/data'
# FILE_PATH = os.path.join(DIR, 'target_dyn_demand.parquet')
# print("📂 Trying to read:", FILE_PATH)
# print("✅ File exists:", os.path.exists(FILE_PATH))


# 경로 설정
MAC_DIR = '../data/'
WINDOW_DIR = 'C:/Users/USER/PycharmProjects/research/data/'

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
    # 1. 데이터 로딩
    # target_dyn_demand = pl.read_parquet(DIR + 'target_dyn_demand.parquet')
    target_dyn_demand = pl.read_parquet(FILE_PATH)
    config = TitanConfigMonthly()

    # 2. 데이터 모듈 구성
    data_module = MultiPartDataModule(
        df=target_dyn_demand,
        config=config,
        batch_size=128,
        val_ratio=0.2
    )

    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()

    # 3. 모델 정의
    model = LMMModel(config)

    # 4. 학습 시작
    trainer = TitanTrain()
    trainer.train_model_with_tta(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )

if __name__ == "__main__":
    main()
