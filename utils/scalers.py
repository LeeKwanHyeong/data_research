from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import polars as pl
import numpy as np

def ts_min_max_scale(series: pl.Series):
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)
    return scaled_series

def ts_min_max_rescale(scaled_series: np.ndarray):
    scaler = MinMaxScaler()



def ts_standard_scale(series: pl.Series):
    scaler = StandardScaler()
    scaled_series = scaler.fit_transform(series)
    return scaled_series

def ts_robust_scaler(series: pl.Series):
    scaler = RobustScaler()
    scaled_series = scaler.fit_transform(series)
    return scaled_series

def ts_max_abs_scaler(series: pl.Series):
    scaler = MaxAbsScaler()
    scaled_series = scaler.fit_transform(series)
    return scaled_series

