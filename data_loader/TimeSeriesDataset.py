import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

