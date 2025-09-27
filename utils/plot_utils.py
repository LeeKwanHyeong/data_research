import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from training.forecaster import DMSForecaster, IMSForecaster
from utils.infer import predict_120_for_model


@torch.no_grad()
def _to_1d_history(x: torch.Tensor) -> np.ndarray:
    """
    x 한 샘플에서 과거 시계열(lookback)을 1D로 뽑는다.
    추정 규칙:
      - (1, L)            -> (L,)
      - (1, L, C)         -> 마지막 채널 또는 첫 채널을 사용(여기선 첫 채널)
      - (1, C, L)         -> 마지막 축이 시간이라고 보고 (L,)
    그 외 모양이면 빈 배열 반환.
    """
    x = x.squeeze(0)
    if x.dim() == 1:             # (L,)
        return x.cpu().numpy()
    if x.dim() == 2:
        # (L, C) or (C, L) 둘 다 있을 수 있어 헷갈림
        # 시간축이 긴 쪽을 시간으로 간주
        h, w = x.shape
        if h >= w:
            # (L, C) 가정 → 첫 채널 사용
            return x[:, 0].cpu().numpy()
        else:
            # (C, L) 가정 → 첫 채널의 시계열
            return x[0, :].cpu().numpy()
    return np.array([])

@torch.no_grad()
def _predict_any(model, x, device="cpu"):
    """
    모델 출력 유형을 자동 처리:
      - (B, H)            -> point
      - (B, C, H)         -> C==1이면 squeeze, 아니면 첫 채널
      - (B, Q, H)         -> quantiles로 간주 (Q==3이면 (0.1,0.5,0.9) 가정)
    반환: dict
      {"point": (H,),
       "q": {"q10":(H,), "q50":(H,), "q90":(H,)}  # 있으면
      }
    """
    out = model(x.to(device))
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.dim() == 2:  # (B, H)
        return {"point": out.squeeze(0).detach().cpu().numpy()}
    if out.dim() == 3:  # (B, A, H)
        B, A, H = out.shape
        arr = out.squeeze(0).detach().cpu().numpy()  # (A, H)
        # A==1 -> point
        if A == 1:
            return {"point": arr[0]}
        # A==3 -> 0.1/0.5/0.9 가정
        if A == 3:
            return {"point": arr[1], "q": {"q10": arr[0], "q50": arr[1], "q90": arr[2]}}
        # 그 외엔 중앙 인덱스를 point로
        mid = A // 2
        return {"point": arr[mid]}
    # (B, nvars, H) 같은 케이스
    if out.dim() == 3 and out.size(1) == 1:
        return {"point": out[:, 0, :].squeeze(0).detach().cpu().numpy()}
    # 지원 외 형태
    y = out.squeeze().detach().cpu().numpy()
    if y.ndim == 1:
        return {"point": y}
    raise RuntimeError(f"Unsupported model output shape: {tuple(out.shape)}")

def _align_len(yhat: np.ndarray, H: int):
    """예측 길이를 H에 맞춤: 1이면 복제, 더 길면 자름, 더 짧으면 NaN 패딩"""
    yhat = np.asarray(yhat).reshape(-1)
    if yhat.size == H:
        return yhat, None
    if yhat.size == 1:
        return np.repeat(yhat, H), "[rep]"
    if yhat.size > H:
        return yhat[:H], "[cut]"
    # 1 < size < H
    pad = np.full(H - yhat.size, np.nan)
    return np.concatenate([yhat, pad], axis=0), "[pad]"

@torch.no_grad()
def _safe_forward(model, x, future_exo=None, mode="eval"):
    """
    모델 시그니처가 다 달라도 안전하게 호출을 시도한다.
    우선 (x)로 호출 → 안되면 (x, future_exo=...) → 안되면 (x, mode=...) → 마지막 (x, future_exo=..., mode=...).
    """
    # 1) model(x)
    try:
        return model(x)
    except TypeError:
        pass
    # 2) model(x, future_exo=...)
    try:
        return model(x, future_exo=future_exo)
    except TypeError:
        pass
    # 3) model(x, mode=...)
    try:
        return model(x, mode=mode)
    except TypeError:
        pass
    # 4) model(x, future_exo=..., mode=...)
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
def plot_val_per_part(models: dict, val_loader, out_dir, device="cpu", max_plots=None):
    """
    models: {"PatchMixer Base": model, ...}
    val_loader: (x, y, part_id) or (x, y)
    - 과거 입력창(x) + 미래 실제(y_true) + 각 모델 예측을 한 그림에.
    - 분위수 모델은 0.1~0.9 음영, 0.5 중앙선.
    """
    os.makedirs(out_dir, exist_ok=True)
    seen = set()
    n_plots = 0

    for batch in val_loader:
        if len(batch) == 3:
            xb, yb, part_ids = batch
        else:
            xb, yb = batch
            part_ids = [f"idx{i}" for i in range(xb.size(0))]

        B = xb.size(0)
        for i in range(B):
            pid = part_ids[i]
            if pid in seen:
                continue
            seen.add(pid)

            x = xb[i:i+1]
            y_true = yb[i:i+1].cpu().numpy().reshape(-1)     # (H,)
            H = y_true.shape[0]

            # --- 과거 히스토리(lookback) ---
            hist = _to_1d_history(x)                         # (L,) or empty
            t_hist = np.arange(-len(hist)+1, 1) if hist.size > 0 else None
            t_fut  = np.arange(1, H+1)

            # --- 모델 예측 ---
            preds_point = {}
            preds_q10, preds_q50, preds_q90 = {}, {}, {}
            for name, model in models.items():
                p = _predict_any(model, x, device=device)
                # point
                yh, tag = _align_len(p["point"], H)
                if tag: name_plot = f"{name} {tag}"
                else:   name_plot = name
                preds_point[name_plot] = yh
                # quantiles 있으면 저장
                if "q" in p:
                    qd = p["q"]
                    q10, _ = _align_len(qd.get("q10", qd.get("p10", qd.get("low", yh))), H)
                    q50, _ = _align_len(qd.get("q50", qd.get("p50", yh)), H)
                    q90, _ = _align_len(qd.get("q90", qd.get("p90", qd.get("high", yh))), H)
                    preds_q10[name_plot] = q10
                    preds_q50[name_plot] = q50
                    preds_q90[name_plot] = q90

            # --- 플롯 ---
            import matplotlib.pyplot as plt
            plt.figure(figsize=(11, 5))

            # 과거
            if hist.size > 0:
                plt.plot(t_hist, hist, label="History", linewidth=2, alpha=0.8)

            # 미래 실제
            plt.plot(t_fut, y_true, label="True", linewidth=2)

            # 분위수 밴드(있을 때)
            for nm in preds_q10.keys():
                q10 = preds_q10[nm]; q50 = preds_q50[nm]; q90 = preds_q90[nm]
                plt.fill_between(t_fut, q10, q90, alpha=0.15, label=f"{nm} P10–P90")
                plt.plot(t_fut, q50, linewidth=1.8, alpha=0.9, label=f"{nm} P50")

            # 포인트 예측(나머지 모델)
            for nm, yhat in preds_point.items():
                # 분위수 중앙선을 이미 그렸다면 중복 라벨 방지
                if nm in preds_q50:
                    continue
                plt.plot(t_fut, yhat, label=nm, alpha=0.9)

            all_points = []
            for nm in preds_point.keys():
                if nm in preds_q90:  # 분위수 모델: 중앙선(P50)을 포인트로 사용
                    base = preds_q90[nm] * 1.5
                else:
                    base = preds_point[nm]
                all_points.append(base)

            if len(all_points) > 0:
                M = np.vstack(all_points)  # (num_models, H)
                ens_mean = np.nanmean(M, axis=0)  # 시점별 평균
                plt.plot(t_fut, ens_mean, linewidth=2.5, alpha=0.95, label="Ensemble mean")

            plt.axvline(0, color="gray", linewidth=1, alpha=0.6)  # 경계선(과거/미래)
            plt.title(f"Validation Forecasts – part: {pid}")
            plt.xlabel("Time (history ≤ 0 < future)")
            plt.ylabel("Demand")
            plt.legend(ncol=2)
            plt.tight_layout()

            fpath = os.path.join(out_dir, f"val_{pid}.png")
            # 저장하고 화면에도 볼 거면 둘 다:
            # plt.savefig(fpath, dpi=150)
            plt.show()
            plt.close()

            n_plots += 1
            if (max_plots is not None) and (n_plots >= max_plots):
                return

@torch.no_grad()
def _probe_output(model, x1, device="cpu", future_exo_cb=None):
    """
    모델 한 번만 호출해서 출력 타입/길이 확인.
    exo가 필요하면 model.horizon 추정 뒤 exo를 만들어서 전달.
    """
    model = model.to(device).eval()

    # 1) 먼저 exo 없이 시도
    try:
        out = _safe_forward(model, x1.to(device), future_exo=None)
    except Exception:
        # 2) 실패하면 horizon 추정 → exo 생성해서 시도
        Hm = _infer_horizon(model, default=120)
        exo = None
        if future_exo_cb is not None:
            exo = future_exo_cb(0, Hm, device=device)  # (H, D)
            exo = exo.unsqueeze(0).expand(x1.size(0), -1, -1)
        out = _safe_forward(model, x1.to(device), future_exo=exo)

    if isinstance(out, (tuple, list)):
        for t in out:
            if torch.is_tensor(t):
                out = t
                break
    return out  # Tensor

@torch.no_grad()
def _predict_120_any(model, x1, device="cpu", future_exo_cb=None):
    """
    모델에 따라 120개월 예측을 얻는다.
    - (B,Q,Hm) 형태(분위수)면 q10/q50/q90를 반환
    - 그 외(포인트형)면 DMS로 120개월 point 예측
    """
    out = _probe_output(model, x1, device=device, future_exo_cb=future_exo_cb)

    # ---- 분위수 모델 ----
    if out.dim() == 3 and out.size(1) >= 3:   # (B,Q,Hm)
        arr = out.squeeze(0).detach().cpu().numpy()   # (Q,Hm)
        q10 = arr[0]; q50 = arr[1]; q90 = arr[2]
        Hm = q50.shape[-1]; H = 120
        def _fix_len(a):
            if a.shape[-1] == H: return a
            if a.shape[-1] >  H: return a[:H]
            pad = np.full(H - a.shape[-1], np.nan)
            return np.concatenate([a, pad], 0)
        q10, q50, q90 = _fix_len(q10), _fix_len(q50), _fix_len(q90)
        return {"point": q50, "q": {"q10": q10, "q50": q50, "q90": q90}}

    # ---- 포인트형 모델: DMS로 120개월 ----
    f = DMSForecaster(
        model,
        target_channel=0,
        fill_mode="copy_last",
        lmm_mode="eval",
        predict_fn=None,
        ttm=None,
        future_exo_cb=future_exo_cb  # 콜백 연결
    )
    # y120 = f.forecast(
    #     x1,
    #     horizon=120,
    #     device=device,
    #     extend='ims',
    #     context_policy='once',
    #     # 필요 시 안정화 옵션도 여기서 끌 수 있음
    #     # use_winsor=False, use_multi_guard=False, use_dampen=False
    # )  # (1,120)
    y120 = f.forecast_overlap_avg(
        x1,
        horizon = 120,
        device = device,
        context_policy = 'once'
    )
    return {"point": y120.squeeze(0).detach().cpu().numpy()}

@torch.no_grad()
def plot_120_months_many(models: dict,
                         val_loader,
                         device="cpu",
                         use_truth=True,
                         max_plots=100,
                         out_dir=None,
                         show=True,
                         future_exo_cb=None):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # probe 유지
    probe_batch = next(iter(val_loader))
    xb_probe = probe_batch[0]
    if xb_probe.dim() == 2:
        xb_probe = xb_probe.unsqueeze(-1)
    x_probe = xb_probe[0:1].to(device)

    plotted = 0
    for batch in val_loader:
        if len(batch) == 3:
            xb, yb, part_ids = batch
        else:
            xb, yb = batch
            part_ids = [f"idx{i}" for i in range(xb.size(0))]

        if xb.dim() == 2:
            xb = xb.unsqueeze(-1)  # (B,L)->(B,L,1)

        B = xb.size(0)
        for i in range(B):
            if plotted >= max_plots:
                return

            x1 = xb[i:i+1].to(device)  # (1,L,1)
            pid = part_ids[i] if i < len(part_ids) else f"idx{i}"

            # GT (있으면)
            y_true = None
            if use_truth and (yb is not None):
                y_true = yb[i:i+1].cpu().numpy().reshape(-1)

            # ---- 각 모델 120개월 예측 수집 ----
            preds_point = {}
            preds_q10, preds_q50, preds_q90 = {}, {}, {}

            for name, mdl in models.items():
                p = _predict_120_any(mdl, x1, device=device, future_exo_cb=future_exo_cb)
                preds_point[name] = p["point"]
                if "q" in p:
                    preds_q10[name] = p["q"]["q10"]
                    preds_q50[name] = p["q"]["q50"]
                    preds_q90[name] = p["q"]["q90"]

            # ---- 히스토리 & 축 ----
            hist = _to_1d_history(x1)  # (L,)
            t_hist = np.arange(-len(hist)+1, 1) if hist.size > 0 else None
            H = 120
            t_fut = np.arange(1, H+1)

            # ---- 플롯 ----
            plt.figure(figsize=(12, 5))
            if hist.size > 0:
                plt.plot(t_hist, hist, label="History", linewidth=2, alpha=0.8)

            if use_truth and (y_true is not None) and (y_true.size > 0):
                yt = y_true[:H] if y_true.size >= H else np.concatenate([y_true, np.full(H - y_true.size, np.nan)])
                plt.plot(t_fut, yt, label="True (val)", linewidth=2)

            # 1) 분위수 밴드 + 중앙선 (있는 모델만)
            for nm in list(preds_q50.keys()):
                q10 = preds_q10[nm]; q50 = preds_q50[nm]; q90 = preds_q90[nm]
                plt.fill_between(t_fut, q10, q90, alpha=0.15, label=f"{nm} P10–P90")
                plt.plot(t_fut, q50, linewidth=1.8, alpha=0.95, label=f"{nm} P50")

            # 2) 포인트 모델(혹은 분위수 없는 모델)의 라인
            for nm, yhat in preds_point.items():
                if nm in preds_q50:
                    continue
                plt.plot(t_fut, yhat, label=nm, alpha=0.9)

            # 3) 최종 앙상블: “q90 기반”
            stack_for_ens = []
            for nm in preds_point.keys():
                base = preds_q90[nm] if nm in preds_q90 else preds_point[nm]
                stack_for_ens.append(base)
            if stack_for_ens:
                M = np.vstack(stack_for_ens)
                ens_q90_mean = np.nanmean(M, axis=0)
                plt.plot(t_fut, ens_q90_mean, linewidth=2.8, alpha=0.95, label="Ensemble (q90-based)")

            plt.axvline(0, color="gray", linewidth=1, alpha=0.6)
            plt.title(f"120-month Forecast – part: {pid}")
            plt.xlabel("Time (history ≤ 0 < future)")
            plt.ylabel("Demand")
            plt.legend(ncol=2)
            plt.tight_layout()

            if out_dir:
                fp = os.path.join(out_dir, f"forecast120_{pid}.png")
                plt.savefig(fp, dpi=150)
            if show:
                plt.show()
            else:
                plt.close()

            plotted += 1