import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import polars as pl

class MultiPartInferenceDataset(Dataset):
    def __init__(self, df: pl.DataFrame, lookback: int, horizon: int):
        self.inputs = []
        self.part_ids = []

        grouped = df.partition_by('oper_part_no')
        for g in grouped:
            series = g.sort('demand_dt')['demand_qty'].to_numpy()
            part_no = g['oper_part_no'][0]

            if len(series) < lookback:
                continue

            x_seq = series[-lookback:]
            self.inputs.append(x_seq)
            self.part_ids.append(part_no)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x_seq = self.inputs[idx]
        part_no = self.part_ids[idx]

        return (
            torch.tensor(x_seq, dtype = torch.float32).unsqueeze(-1),
            part_no
        )

class MultiPartTrainingDataset(Dataset):
    def __init__(self, df: pl.DataFrame, lookback: int, horizon: int):
        self.samples = []
        self.part_ids = []

        grouped = df.partition_by('oper_part_no')
        for g in grouped:
            series = g['demand_qty'].to_numpy()
            part_no = g['oper_part_no'][0]
            if len(series) < lookback + horizon:
                continue
            for i in range(len(series) - lookback - horizon + 1):
                x_seq = series[i: i+lookback]
                y_seq = series[i+lookback: i+lookback+horizon]
                self.samples.append((x_seq, y_seq))
                self.part_ids.append(part_no)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, y_seq = self.samples[idx]
        return (
            torch.tensor(x_seq, dtype = torch.float32).unsqueeze(-1),
            torch.tensor(y_seq, dtype = torch.float32),
            self.part_ids[idx]
        )

class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback, horizon):
        '''
            data: 1D numpy array or list (time series)
            lookback: input sequence length
            horizon: prediction length
        '''
        self.data = data
        self.lookback = lookback
        self.horizon = horizon
        self.samples = []

        for i in range(len(data) - lookback - horizon):
            x_seq = data[i:i + lookback]
            y_seq = data[i + lookback: i + lookback + horizon]
            self.samples.append((x_seq, y_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, y_seq = self.samples[idx]
        return torch.tensor(x_seq, dtype = torch.float32).unsqueeze(-1), \
               torch.tensor(y_seq, dtype = torch.float32) # [lookback, 1], [horizon]

