import torch
import numpy as np
from modeling_module.training.forecaster import DMSForecaster, IMSForecaster


@torch.no_grad()
def predict_standardized(model, x, device="cpu", q_index=None):
    """
    x: (B, lookback, C=1) 또는 데이터모듈 출력 형태.
    반환: point 예측 (B, H)
    - (B,Q,H) 형태면 Q의 0.5(또는 중앙 인덱스)를 골라 point로 변환
    """
    if q_index is None:
        q_index = {0.1: 0, 0.5: 1, 0.9: 2}
    x = x.to(device)
    out = model(x)  # adapters가 있으면 그 경유로 호출
    if torch.is_tensor(out):
        yhat = out
    elif isinstance(out, (tuple, list)):
        yhat = out[0]
    else:
        raise RuntimeError("Unknown model output")

    # (B, H) or (B, C, H) or (B, Q, H)
    if yhat.dim() == 3:
        B, A, H = yhat.shape
        # Q 또는 C 구분 없이 중앙/0.5 인덱스 사용
        mid = q_index.get(0.5, A // 2)
        yhat = yhat[:, mid, :]
    elif yhat.dim() == 2:
        pass
    else:
        # (B, nvars, H) → 단변량이면 squeeze
        if yhat.dim() == 3 and yhat.size(1) == 1:
            yhat = yhat[:, 0, :]
        else:
            raise RuntimeError(f"Unsupported output shape: {tuple(yhat.shape)}")
    return yhat  # (B, H)

# --- 도우미: 모델 출력이 (B,Q,H)이면 P50만 뽑아 point로 바꿔주는 predict_fn 생성 ---
def make_predict_fn_point_from_quantile(model, q_index=1):
    @torch.no_grad()
    def _predict(x):
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        # 기대: (B, Q, H)
        assert out.dim() == 3 and out.size(1) > q_index, f"Unexpected shape for quantile: {tuple(out.shape)}"
        return out[:, q_index, :]  # (B,H)
    return _predict

# --- 어떤 모델이든 120개월 예측을 시도하는 공통 추론기(가능하면 DMS, 아니면 IMS) ---
@torch.no_grad()
def predict_120_for_model(model, x_init, device="cpu", prefer="dms") -> torch.Tensor:
    """
    반환: y_hat (B, 120)
    - 분위수 모델은 중앙(P50)로 point화
    - Titan/LMM은 lmm_mode='eval'로 호출
    """
    model = model.to(device).eval()
    B = x_init.size(0)

    # 샘플 한 번 호출해서 출력형 파악
    try:
        tmp = model(x_init.to(device))
    except TypeError:
        tmp = model(x_init.to(device), mode='eval')

    if isinstance(tmp, (tuple, list)):
        tmp = tmp[0]

    predict_fn = None
    lmm_mode = None

    # (B, Q, H) → 중앙 분위수 사용
    if tmp.dim() == 3 and tmp.size(1) >= 3:
        predict_fn = make_predict_fn_point_from_quantile(model, q_index=1)

    # Titan/LMM인 경우를 대비해 lmm_mode 지정(필요한 모델만 내부에서 사용됨)
    lmm_mode = 'eval'

    # Forecaster 구성
    if prefer == "dms":
        f = DMSForecaster(model, target_channel=0, fill_mode="copy_last",
                          lmm_mode=lmm_mode, predict_fn=predict_fn, ttm=None)
        y_hat = f.point_DMS_to_IMS(x_init, horizon=120, device=device, extend='ims', context_policy='once')
    else:
        f = IMSForecaster(model, target_channel=0, fill_mode="copy_last",
                          lmm_mode=lmm_mode, predict_fn=predict_fn, ttm=None)
        y_hat = f.forecast(x_init, horizon=120, device=device, context_policy='once')

    # 보장: (B,120)
    if y_hat.dim() == 1:
        y_hat = y_hat.view(B, -1)
    return y_hat  # (B,120)

# --- 히스토리 1D 추출 (앞서 만든 함수 그대로 사용) ---
@torch.no_grad()
def _to_1d_history(x: torch.Tensor) -> np.ndarray:
    x = x.squeeze(0)
    if x.dim() == 1:
        return x.cpu().numpy()
    if x.dim() == 2:
        h, w = x.shape
        if h >= w:
            return x[:, 0].cpu().numpy()
        else:
            return x[0, :].cpu().numpy()
    return np.array([])