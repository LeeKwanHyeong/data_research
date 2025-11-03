# plot_utils.py
# -*- coding: utf-8 -*-

import os
from typing import Dict, Tuple, Optional, Callable

import numpy as np
import torch
import matplotlib.pyplot as plt

from modeling_module.training.forecaster import DMSForecaster

# ==============================
# Global DEBUG
# ==============================
DEBUG_FCAST = True


# ==============================
# Small helpers
# ==============================
def _log_var(tag, arr):  # arr: (H,) 또는 (H,1)
    arr = np.asarray(arr).reshape(-1)
    print(f"[PLOT-DBG] {tag}: var(H)={np.nanvar(arr):.6g}, "
          f"mean(H)={np.nanmean(arr):.6g}, unique(H)={np.unique(np.round(arr, 6)).size}")


@torch.no_grad()
def _to_1d_history(x: torch.Tensor) -> np.ndarray:
    """
    단일 샘플 텐서에서 lookback 구간의 1D 시계열을 추출한다.
    기대 입력: x.shape in {(1,L), (1,L,1), (1,L,C), (1,C,L)}
    """
    x = x.squeeze(0)
    if x.dim() == 1:             # (L,)
        return x.detach().cpu().numpy()
    if x.dim() == 2:
        h, w = x.shape
        if h >= w:               # (L, C) 가정 → 첫 채널
            return x[:, 0].detach().cpu().numpy()
        else:                    # (C, L) 가정 → 첫 채널의 시계열
            return x[0, :].detach().cpu().numpy()
    return np.array([])


@torch.no_grad()
def _safe_forward(model, x, future_exo=None, mode="eval"):
    """
    모델 시그니처 차이를 흡수하기 위한 안전 호출.
    우선순위: (x) → (x,future_exo) → (x,mode) → (x,future_exo,mode)
    """
    try:
        return model(x)
    except TypeError:
        pass
    try:
        return model(x, future_exo=future_exo)
    except TypeError:
        pass
    try:
        return model(x, mode=mode)
    except TypeError:
        pass
    return model(x, future_exo=future_exo, mode=mode)


@torch.no_grad()
def _infer_horizon(model, default=120):
    for k in ("horizon", "output_horizon", "H", "Hm"):
        if hasattr(model, k):
            try:
                return int(getattr(model, k))
            except Exception:
                pass
    return default


@torch.no_grad()
def _probe_output(model, x1, device="cpu", future_exo_cb=None):
    """
    모델을 한 번 호출해 '형태 파악용' 출력을 가져온다.
    - 반환은 Tensor 또는 dict( {'point': Tensor, 'q': Tensor|dict} )일 수 있다.
    """
    model = model.to(device).eval()

    def _call(m, x, ex=None):
        try:
            return m(x)
        except TypeError:
            pass
        try:
            return m(x, future_exo=ex)
        except TypeError:
            pass
        try:
            return m(x, mode="eval")
        except TypeError:
            pass
        return m(x, future_exo=ex, mode="eval")

    # 1) exo 없이 우선 시도
    try:
        out = _call(model, x1.to(device), None)
    except Exception:
        # 2) 실패하면 horizon 추정 → exo 포함
        Hm = _infer_horizon(model, default=120)
        exo = None
        if future_exo_cb is not None:
            ex = future_exo_cb(0, Hm, device=device)  # (H, D)
            exo = ex.unsqueeze(0).expand(x1.size(0), -1, -1)  # (B,H,D)
        out = _call(model, x1.to(device), exo)

    # tuple/list면 첫 텐서나 dict를 꺼낸다
    if isinstance(out, (tuple, list)):
        for t in out:
            if torch.is_tensor(t) or isinstance(t, dict):
                out = t
                break
    return out  # Tensor or dict

# ==============================
# Quantile rolling (IMS)
# ==============================
@torch.no_grad()
def _roll_quantile_ims(
    model,
    x_init,
    horizon: int,
    *,
    device: str = "cpu",
    future_exo_cb=None,
    target_channel: int = 0,
    fill_mode: str = "copy_last",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantile 모델(출력 (B,Q,H) 또는 (B,H,Q))을 IMS로 굴려 길이=horizon의
    q10/q50/q90 시퀀스를 만든다. 반환: (q10, q50, q90) 각 (B,H).
    """
    model = model.to(device).eval()
    x = x_init.to(device).float().clone()
    if x.dim() == 2:
        x = x.unsqueeze(-1)  # (B,L)->(B,L,1)
    B = x.size(0)

    # probe로 모양 파악
    out_probe = model(x)
    if isinstance(out_probe, (tuple, list)):
        out_probe = next(t for t in out_probe if torch.is_tensor(t))
    assert out_probe.dim() == 3, f"expect 3D output for quantile model, got {tuple(out_probe.shape)}"

    # 축 자동 감지
    if out_probe.shape[1] in (3, 5, 9):          # (B,Q,Hm)
        Hm = out_probe.shape[2]
        def extract_first_step_q(out):
            if isinstance(out, (tuple, list)):
                out = next(tt for tt in out if torch.is_tensor(tt))
            return out[:, 0, 0], out[:, 1, 0], out[:, 2, 0]
    elif out_probe.shape[2] in (3, 5, 9):        # (B,Hm,Q)
        Hm = out_probe.shape[1]
        def extract_first_step_q(out):
            if isinstance(out, (tuple, list)):
                out = next(tt for tt in out if torch.is_tensor(tt))
            return out[:, 0, 0], out[:, 0, 1], out[:, 0, 2]
    else:
        raise RuntimeError(f"cannot infer quantile axis from shape {tuple(out_probe.shape)}")

    q10_seq, q50_seq, q90_seq = [], [], []

    def _prepare_next_input(x, y_step, *, target_channel=0, fill_mode="copy_last"):
        B, L, C = x.shape
        y_step = y_step.reshape(B, 1, 1)
        if C == 1:
            new_tok = y_step
        else:
            last = x[:, -1:, :].clone()
            new_tok = torch.zeros_like(last) if fill_mode == "zeros" else last
            new_tok[:, 0, target_channel] = y_step[:, 0, 0]
        return torch.cat([x[:, 1:, :], new_tok], dim=1)

    for t in range(int(horizon)):
        # exo 준비 (길이는 항상 Hm)
        exo = None
        if future_exo_cb is not None:
            ex = future_exo_cb(t, Hm, device=device)   # (Hm, D)
            exo = ex.unsqueeze(0).expand(B, -1, -1)    # (B,Hm,D)

        out = model(x, future_exo=exo)
        q10_t, q50_t, q90_t = extract_first_step_q(out)

        if DEBUG_FCAST and t < 5:
            nm = str(getattr(model, "model_name", "Unknown"))
            print(f"[Q-IMS][{nm}] t={t} q10={float(q10_t[0]):.6g} q50={float(q50_t[0]):.6g} q90={float(q90_t[0]):.6g}")

        q10_seq.append(q10_t.unsqueeze(1))
        q50_seq.append(q50_t.unsqueeze(1))
        q90_seq.append(q90_t.unsqueeze(1))

        # q50으로 다음 입력 윈도우 갱신
        x = _prepare_next_input(x, q50_t, target_channel=target_channel, fill_mode=fill_mode)

    q10 = torch.cat(q10_seq, dim=1)  # (B,H)
    q50 = torch.cat(q50_seq, dim=1)
    q90 = torch.cat(q90_seq, dim=1)
    return q10, q50, q90

def _align_len(yhat: np.ndarray, H: int):
    yhat = np.asarray(yhat).reshape(-1)
    if yhat.size == H:
        return yhat, None
    if yhat.size == 1:
        return np.repeat(yhat, H), "[rep]"
    if yhat.size > H:
        return yhat[:H], "[cut]"
    pad = np.full(H - yhat.size, np.nan)
    return np.concatenate([yhat, pad], axis = 0), "[pad]"

# ==============================
# Common predictor for any model
# ==============================
@torch.no_grad()
def _predict_any(model, x1, device="cpu", future_exo_cb=None, horizon: int = 120):
    """
    모델 종류(포인트/분위수)에 관계없이 예측을 표준 딕셔너리로 반환:
      - 포인트: {"point": np.ndarray(H,)}
      - 분위수: {"point": q50, "q": {"q10":..., "q50":..., "q90":...}}
    """
    out = _probe_output(model, x1, device=device, future_exo_cb=future_exo_cb)
    is_q_flag = bool(getattr(model, "is_quantile", False))

    # --- 1) 모델이 dict를 직접 반환하는 경우 ---
    if isinstance(out, dict):
        # (A) 분위수가 들어있는 경우
        if "q" in out:
            q = out["q"]

            # q가 Tensor인 케이스: [B,H,3] 또는 [B,3,H] 등에서 q10,q50,q90 추출
            if torch.is_tensor(q):
                if q.dim() != 3:
                    raise RuntimeError(f"Unsupported q tensor shape: {tuple(q.shape)}")
                B, D1, D2 = q.shape
                if D2 in (3, 5, 9):      # (B,H,Q)
                    q10_t = q[:, :, 0]
                    q50_t = q[:, :, 1]
                    q90_t = q[:, :, 2]
                elif D1 in (3, 5, 9):    # (B,Q,H)
                    q10_t = q[:, 0, :]
                    q50_t = q[:, 1, :]
                    q90_t = q[:, 2, :]
                else:
                    raise RuntimeError(f"Cannot infer quantile axis from shape {tuple(q.shape)}")

                q10 = q10_t.squeeze(0).detach().cpu().numpy()
                q50 = q50_t.squeeze(0).detach().cpu().numpy()
                q90 = q90_t.squeeze(0).detach().cpu().numpy()

            # q가 dict인 케이스: {'q10':Tensor/ndarray, 'q50':..., 'q90':...}
            elif isinstance(q, dict):
                def _to_np(v):
                    if torch.is_tensor(v):
                        return v.squeeze(0).detach().cpu().numpy().reshape(-1)
                    return np.asarray(v, dtype=float).reshape(-1)
                q10 = _to_np(q.get("q10"))
                q50 = _to_np(q.get("q50"))
                q90 = _to_np(q.get("q90"))

            else:
                raise RuntimeError("Unsupported 'q' type in model output dict.")

            # 길이 정렬
            q10, _ = _align_len(q10, horizon)
            q50, _ = _align_len(q50, horizon)
            q90, _ = _align_len(q90, horizon)

            return {
                "point": q50,
                "q": {"q10": q10, "q50": q50, "q90": q90},
            }

        # (B) point만 있는 경우
        if "point" in out:
            pt = out["point"]
            if torch.is_tensor(pt):
                point = pt.squeeze(0).detach().cpu().numpy().reshape(-1)
            else:
                point = np.asarray(pt, dtype=float).reshape(-1)
            point, _ = _align_len(point, horizon)
            return {"point": point}

        # dict지만 포맷을 모르면 첫 텐서를 point로 사용
        for v in out.values():
            if torch.is_tensor(v):
                point = v.squeeze(0).detach().cpu().numpy().reshape(-1)
                point, _ = _align_len(point, horizon)
                return {"point": point}
        raise RuntimeError("Unsupported dict output structure from model.")

    # --- 2) Tensor 출력: 3D라면 분위수 모델로 간주할 수 있음 ---
    if torch.is_tensor(out) and out.dim() == 3:
        B, D1, D2 = out.shape
        has_quant_axis = (D1 in (3, 5, 9)) or (D2 in (3, 5, 9))
        is_quantile = has_quant_axis or is_q_flag

        if is_quantile and has_quant_axis:
            # IMS로  q10/q50/q90 시퀀스 생성
            q10, q50, q90 = _roll_quantile_ims(
                model, x1, horizon=horizon, device=device,
                future_exo_cb=future_exo_cb, target_channel=0, fill_mode="copy_last"
            )
            return {
                "point": q50.squeeze(0).cpu().numpy(),
                "q": {
                    "q10": q10.squeeze(0).cpu().numpy(),
                    "q50": q50.squeeze(0).cpu().numpy(),
                    "q90": q90.squeeze(0).cpu().numpy(),
                },
            }

    # --- 3) 그 외는 포인트형으로 간주 → Forecaster 사용 ---
    f = DMSForecaster(
        model,
        target_channel=0,
        fill_mode="copy_last",
        lmm_mode="eval",
        predict_fn=None,
        ttm=None,
        future_exo_cb=future_exo_cb,
    )
    y_hat = f.forecast_DMS_to_IMS(
        x_init=x1,                 # ← 중요: x_init_raw 로 호출
        horizon=horizon,
        device=device,
        extend="ims",
        context_policy="once",
    )
    return {"point": y_hat.squeeze(0).detach().cpu().numpy()}


# ==============================
# Core plotting
# ==============================
def _plot_single_series(
    *,
    hist: Optional[np.ndarray],
    y_true: Optional[np.ndarray],
    preds_point: Dict[str, np.ndarray],
    preds_q10: Dict[str, np.ndarray],
    preds_q50: Dict[str, np.ndarray],
    preds_q90: Dict[str, np.ndarray],
    horizon: int,
    title: str,
    out_path: Optional[str],
    show: bool,
    zoom_future: bool = False,
    zoom_len: Optional[int] = None,
):
    """
    한 파트에 대해 히스토리 + 여러 모델의 예측을 그린다.
    - horizon 길이를 기준으로 x-축(1..H)을 맞춘다.
    - zoom_future=True 이면 미래 구간 일부만(예: 27) 확대하여 그린다.
    """
    t_hist = np.arange(-len(hist) + 1, 1) if (hist is not None and hist.size > 0) else None
    t_fut = np.arange(1, horizon + 1)

    plt.figure(figsize=(12, 5))

    # history
    if hist is not None and hist.size > 0:
        plt.plot(t_hist, hist, label="History", linewidth=2, alpha=0.8)

    # ground truth (있다면)
    if y_true is not None:
        yt = np.asarray(y_true, float).reshape(-1)
        if yt.size > horizon:
            yt = yt[:horizon]
        elif yt.size < horizon:
            yt = np.concatenate([yt, np.full(horizon - yt.size, np.nan)])
        if zoom_future:
            zL = int(zoom_len or horizon)
            zL = max(1, min(zL, horizon))
            plt.plot(t_fut[:zL], yt[:zL], label="True", linewidth=2)
        else:
            plt.plot(t_fut, yt, label="True", linewidth=2)

    # quantile (있다면)
    for nm in list(preds_q50.keys()):
        q10 = np.asarray(preds_q10.get(nm))
        q50 = np.asarray(preds_q50.get(nm))
        q90 = np.asarray(preds_q90.get(nm))
        if q10 is None or q50 is None or q90 is None:
            continue

        def _fit(a):
            a = a.reshape(-1)
            if a.size > horizon:
                return a[:horizon]
            if a.size < horizon:
                return np.concatenate([a, np.full(horizon - a.size, np.nan)])
            return a

        q10, q50, q90 = _fit(q10), _fit(q50), _fit(q90)
        if zoom_future:
            zL = int(zoom_len or horizon); zL = max(1, min(zL, horizon))
            plt.fill_between(t_fut[:zL], q10[:zL], q90[:zL], alpha=0.15, label=f"{nm} P10–P90")
            plt.plot(t_fut[:zL], q50[:zL], linewidth=1.8, alpha=0.95, label=f"{nm} P50")
        else:
            plt.fill_between(t_fut, q10, q90, alpha=0.15, label=f"{nm} P10–P90")
            plt.plot(t_fut, q50, linewidth=1.8, alpha=0.95, label=f"{nm} P50")

    # point-only models
    for nm, yhat in preds_point.items():
        if nm in preds_q50:  # 중앙선 중복 회피
            continue
        a = np.asarray(yhat).reshape(-1)
        if a.size > horizon:
            a = a[:horizon]
        elif a.size < horizon:
            a = np.concatenate([a, np.full(horizon - a.size, np.nan)])
        if zoom_future:
            zL = int(zoom_len or horizon); zL = max(1, min(zL, horizon))
            plt.plot(t_fut[:zL], a[:zL], label=nm, alpha=0.9)
        else:
            plt.plot(t_fut, a, label=nm, alpha=0.9)

    # 간단 앙상블 (q90 기반)
    stack = []
    for nm in preds_point.keys():
        base = preds_q90.get(nm, preds_point[nm])
        base = np.asarray(base).reshape(-1)
        if base.size > horizon:
            base = base[:horizon]
        elif base.size < horizon:
            base = np.concatenate([base, np.full(horizon - base.size, np.nan)])
        stack.append(base)
    if stack:
        M = np.vstack(stack)
        ens_q90 = np.nanmean(M, axis=0)
        if zoom_future:
            zL = int(zoom_len or horizon); zL = max(1, min(zL, horizon))
            plt.plot(t_fut[:zL], ens_q90[:zL], linewidth=2.8, alpha=0.95, label="Ensemble (q90-based)")
        else:
            plt.plot(t_fut, ens_q90, linewidth=2.8, alpha=0.95, label="Ensemble (q90-based)")

    plt.axvline(0, color="gray", linewidth=1, alpha=0.6)
    plt.title(title)
    plt.xlabel("Time (history ≤ 0 < future)")
    plt.ylabel("Demand")
    plt.legend(ncol=2)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


# ==============================
# Unified executors (VAL / INFER)
# ==============================
@torch.no_grad()
def _run_and_plot_many(
    *,
    models: Dict[str, torch.nn.Module],
    loader,
    device: str = "cuda" if torch.cuda.is_available() else 'cpu',
    horizon: int,
    mode: str,                      # 'val' | 'infer'
    plan_dt: Optional[int] = None,  # anchor label (YYYYMM or YYYYWW)
    granularity: str = "month",     # 'month' | 'week'
    max_plots: int = 100,
    out_dir: Optional[str] = None,
    show: bool = True,
    future_exo_cb=None,
    truth_cb: Optional[Callable[[str, int, int, str], Optional[np.ndarray]]] = None,
    zoom_future: bool = False,
    zoom_len: Optional[int] = None,
):
    """
    단일 엔진:
      - mode='val'  : (xb, yb[, part_ids]) 배치에서 y_true와 함께 플롯
      - mode='infer': (xb, part_ids) 배치에서 히스토리 + 예측만 플롯(원하면 truth_cb로 GT 조회 가능)
    plan_dt가 있으면 타이틀에 앵커 표기.
    """
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plotted = 0
    for batch in loader:
        if mode == "val":
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("val loader batch must be (xb, yb[, part_ids]).")
            xb, yb = batch[0], batch[1]
            part_ids = batch[2] if len(batch) >= 3 else None
        else:  # 'infer'
            if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                raise ValueError("inference loader batch must be (xb, part_ids).")
            xb, part_ids = batch
            yb = None

        if xb.dim() == 2:
            xb = xb.unsqueeze(-1)  # (B,L)->(B,L,1)

        B = xb.size(0)
        for i in range(B):
            if plotted >= max_plots:
                return
            x1 = xb[i:i+1].to(device)
            pid = (part_ids[i] if (part_ids is not None and i < len(part_ids))
                   else f"idx{i}")

            # y_true 준비
            y_true = None
            if mode == "val" and (yb is not None):
                y_true = yb[i:i+1].detach().cpu().numpy().reshape(-1)
                if y_true.size > horizon:
                    y_true = y_true[:horizon]
            elif (mode == "infer") and (truth_cb is not None) and (plan_dt is not None):
                y_true = truth_cb(pid, plan_dt, horizon, granularity)
                if y_true is not None:
                    y_true = np.asarray(y_true, float).reshape(-1)
                    if y_true.size > horizon:
                        y_true = y_true[:horizon]
                    elif y_true.size < horizon:
                        y_true = np.concatenate([y_true, np.full(horizon - y_true.size, np.nan)])

            # 각 모델 예측 수집
            preds_point, preds_q10, preds_q50, preds_q90 = {}, {}, {}, {}
            for name, mdl in models.items():
                p = _predict_any(mdl, x1, device=device, future_exo_cb=future_exo_cb, horizon=horizon)
                preds_point[name] = p["point"]
                if "q" in p:
                    preds_q10[name] = p["q"].get("q10")
                    preds_q50[name] = p["q"].get("q50")
                    preds_q90[name] = p["q"].get("q90")

            # DEBUG
            if DEBUG_FCAST and horizon == 27:
                for nm, yhat in preds_point.items():
                    _log_var(f"{nm} point(H=27)", yhat[:27])

            # 타이틀
            if plan_dt is not None:
                title = f"[{mode.upper()}:{granularity}] H={horizon} from {plan_dt} – part: {pid}"
            else:
                title = f"[{mode.upper()}] H={horizon} – part: {pid}"

            # 플롯
            hist = _to_1d_history(x1)
            out_path = (os.path.join(out_dir, f"{mode}_{granularity}_H{horizon}_{pid}.png")
                        if out_dir else None)
            _plot_single_series(
                hist=hist,
                y_true=y_true,
                preds_point=preds_point,
                preds_q10=preds_q10,
                preds_q50=preds_q50,
                preds_q90=preds_q90,
                horizon=horizon,
                title=title,
                out_path=out_path,
                show=show,
                zoom_future=zoom_future,
                zoom_len=zoom_len,
            )
            plotted += 1


# ==============================
# Public API (27-week / 120-month)
# ==============================
@torch.no_grad()
def plot_27w(
    models: Dict[str, torch.nn.Module],
    loader,
    *,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    mode: str = "val",                # 'val' | 'infer'
    plan_yyyyww: Optional[int] = None,
    max_plots: int = 100,
    out_dir: Optional[str] = None,
    show: bool = True,
    future_exo_cb=None,
    truth_cb: Optional[Callable[[str, int, int, str], Optional[np.ndarray]]] = None,
):
    """
    27주 예측 전용 플로터.
      - mode='val'이면 (xb, yb[, part_ids]) 배치에서 y_true와 함께 그림.
      - mode='infer'이면 (xb, part_ids) 배치에서 히스토리 + 예측만 표시
        (원하면 truth_cb(part_id, plan_yyyyww, 27, 'week')로 GT 조회).
    """
    _run_and_plot_many(
        models=models,
        loader=loader,
        device=device,
        horizon=27,
        mode=mode,
        plan_dt=plan_yyyyww,
        granularity="week",
        max_plots=max_plots,
        out_dir=out_dir,
        show=show,
        future_exo_cb=future_exo_cb,
        truth_cb=truth_cb,
        zoom_future=True,
        zoom_len=27,
    )


@torch.no_grad()
def plot_120m(
    models: Dict[str, torch.nn.Module],
    loader,
    *,
    device: str = "cpu",
    mode: str = "val",                # 'val' | 'infer'
    plan_yyyymm: Optional[int] = None,
    max_plots: int = 100,
    out_dir: Optional[str] = None,
    show: bool = True,
    future_exo_cb=None,
    truth_cb: Optional[Callable[[str, int, int, str], Optional[np.ndarray]]] = None,
):
    """
    120개월 예측 전용 플로터.
      - mode='val'이면 (xb, yb[, part_ids]) 배치에서 y_true와 함께 그림.
      - mode='infer'이면 (xb, part_ids) 배치에서 히스토리 + 예측만 표시
        (원하면 truth_cb(part_id, plan_yyyymm, 120, 'month')로 GT 조회).
    """
    _run_and_plot_many(
        models=models,
        loader=loader,
        device=device,
        horizon=120,
        mode=mode,
        plan_dt=plan_yyyymm,
        granularity="month",
        max_plots=max_plots,
        out_dir=out_dir,
        show=show,
        future_exo_cb=future_exo_cb,
        truth_cb=truth_cb,
        zoom_future=False,   # 월 120은 전체 보기 기본
        zoom_len=None,
    )


# ==============================
# Optional: simple calendar exo
# ==============================
def make_calendar_exo(start_idx: int, H: int, *, period: int = 52, device: str | torch.device = "cpu") -> torch.Tensor:
    """
    단순 주기성(sin/cos) 외생변수 생성: (H, 2)
    """
    t = torch.arange(start_idx, start_idx + H, device=device, dtype=torch.float32)
    exo = torch.stack([torch.sin(2 * torch.pi * t / period),
                       torch.cos(2 * torch.pi * t / period)], dim=-1)  # (H, 2)
    return exo
