import torch

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