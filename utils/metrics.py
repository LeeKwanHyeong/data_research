import numpy as np

def rse(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def corr(pred, true):
    u = ((true - true.mean()) * (pred - pred.mean())).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u/d).mean(-1)

def mae(pred, true):
    return np.mean(np.abs(pred - true))

def mse(pred, true):
    return np.mean((pred - true) ** 2)

def rmse(pred, true):
    return np.sqrt(mse(pred, true))

def mape(pred, true):
    return np.mean(np.abs((pred - true) / true))

def mspe(pred, true):
    return np.mean(np.square((pred - true) / true))

def smape(y, yhat, eps=1e-8):
    return float(np.mean(2.0 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + eps)))

def metric(pred, true):
    o_mae = mae(pred, true)
    o_mse = mse(pred, true)
    o_rmse = rmse(pred, true)
    o_mape = mape(pred, true)
    o_mspe = mspe(pred, true)
    o_smape = smape(pred, true)
    o_rse = rse(pred, true)
    o_corr = corr(pred, true)
    return o_mae, o_mse, o_rmse, o_mape, o_mspe, o_smape, o_rse, o_corr


