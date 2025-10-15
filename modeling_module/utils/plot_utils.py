import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from modeling_module.training.forecaster import DMSForecaster


# -------------------------
# helpers
# -------------------------
@torch.no_grad()
def _to_1d_history(x: torch.Tensor) -> np.ndarray:
    """
    x 한 샘플에서 과거 시계열(lookback)을 1D로 뽑는다.
    """
    x = x.squeeze(0)
    if x.dim() == 1:             # (L,)
        return x.cpu().numpy()
    if x.dim() == 2:
        h, w = x.shape
        if h >= w:               # (L, C) 가정 → 첫 채널
            return x[:, 0].cpu().numpy()
        else:                    # (C, L) 가정 → 첫 채널의 시계열
            return x[0, :].cpu().numpy()
    return np.array([])


@torch.no_grad()
def _safe_forward(model, x, future_exo=None, mode="eval"):
    """
    모델 시그니처가 달라도 안전 호출:
    (x) → (x,future_exo) → (x,mode) → (x,future_exo,mode)
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


def _align_len(yhat: np.ndarray, H: int):
    """
    예측 길이를 H에 맞춤: 1이면 복제, 더 길면 자름, 더 짧으면 NaN 패딩
    """
    yhat = np.asarray(yhat).reshape(-1)
    if yhat.size == H:
        return yhat, None
    if yhat.size == 1:
        return np.repeat(yhat, H), "[rep]"
    if yhat.size > H:
        return yhat[:H], "[cut]"
    pad = np.full(H - yhat.size, np.nan)
    return np.concatenate([yhat, pad], axis=0), "[pad]"


@torch.no_grad()
def _probe_output(model, x1, device="cpu", future_exo_cb=None):
    """
    모델 한 번만 호출해서 출력 텐서 획득.
    exo가 필요하면 model.horizon 추정 뒤 exo 생성해서 전달.
    """
    model = model.to(device).eval()

    # 1) exo 없이 시도
    try:
        out = _safe_forward(model, x1.to(device), future_exo=None)
    except Exception:
        # 2) 실패하면 horizon 추정 → exo 생성해서 시도
        Hm = _infer_horizon(model, default=120)
        exo = None
        if future_exo_cb is not None:
            exo = future_exo_cb(0, Hm, device=device)  # (H, D)
            exo = exo.unsqueeze(0).expand(x1.size(0), -1, -1)  # (B,H,D)
        out = _safe_forward(model, x1.to(device), future_exo=exo)

    if isinstance(out, (tuple, list)):
        for t in out:
            if torch.is_tensor(t):
                out = t
                break
    return out  # Tensor


@torch.no_grad()
def _roll_quantile_ims(model, x_init, horizon, device="cpu", future_exo_cb=None,
                       target_channel=0, fill_mode="copy_last"):
    """
    Quantile 모델(출력 (B,Q,H) 또는 (B,H,Q))을 IMS로 굴려 길이=horizon의
    q10/q50/q90 시퀀스를 만든다.
    """
    model = model.to(device).eval()
    x = x_init.to(device).float().clone()
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    B = x.size(0)

    # probe: 출력 모양 파악
    out_probe = model(x)
    if isinstance(out_probe, (tuple, list)):
        out_probe = next(t for t in out_probe if torch.is_tensor(t))
    assert out_probe.dim() == 3, f"expect 3D output for quantile model, got {tuple(out_probe.shape)}"

    # ---- 축 자동 감지 ----
    if out_probe.shape[1] in (3, 5, 9):          # (B,Q,Hm)
        Q_axis, H_axis = 1, 2
        Hm = out_probe.shape[H_axis]
        def extract_first_step_q(out):
            if isinstance(out, (tuple, list)):
                out = next(tt for tt in out if torch.is_tensor(tt))
            # (B,Q,H) 가정
            return out[:, 0, 0], out[:, 1, 0], out[:, 2, 0]  # q10,q50,q90 @ step0
    elif out_probe.shape[2] in (3, 5, 9):        # (B,Hm,Q)
        Q_axis, H_axis = 2, 1
        Hm = out_probe.shape[H_axis]
        def extract_first_step_q(out):
            if isinstance(out, (tuple, list)):
                out = next(tt for tt in out if torch.is_tensor(tt))
            # (B,H,Q) 가정
            return out[:, 0, 0], out[:, 0, 1], out[:, 0, 2]  # q10,q50,q90 @ step0
    else:
        raise RuntimeError(f"cannot infer quantile axis from shape {tuple(out_probe.shape)}")

    q10_seq, q50_seq, q90_seq = [], [], []

    def _prepare_next_input(x, y_step, target_channel=0, fill_mode="copy_last"):
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
        # exo 준비 (길이는 Hm로)
        exo = None
        if future_exo_cb is not None:
            ex = future_exo_cb(t, Hm, device=device)   # (Hm, D)
            exo = ex.unsqueeze(0).expand(B, -1, -1)    # (B,Hm,D)

        out = model(x, future_exo=exo)
        q10_t, q50_t, q90_t = extract_first_step_q(out)
        model_name = str(getattr(model, "model_name", "None"))

        if t < 5:  # DEBUG
            print('model', model_name,"[DBG] t", t,
                  "q10:", float(q10_t[0]), "q50:", float(q50_t[0]), "q90:", float(q90_t[0]))
        q10_seq.append(q10_t.unsqueeze(1))
        q50_seq.append(q50_t.unsqueeze(1))
        q90_seq.append(q90_t.unsqueeze(1))

        # q50으로 다음 입력 윈도우 갱신
        x = _prepare_next_input(x, q50_t, target_channel=target_channel, fill_mode=fill_mode)

    q10 = torch.cat(q10_seq, dim=1)  # (B,H)
    q50 = torch.cat(q50_seq, dim=1)
    q90 = torch.cat(q90_seq, dim=1)
    return q10, q50, q90

# -------------------------
# 120개월 예측 (모든 모델 공통)
# -------------------------
@torch.no_grad()
def _predict_120_any(model, x1, device="cpu", future_exo_cb=None):
    out = _probe_output(model, x1, device=device, future_exo_cb=future_exo_cb)

    is_q_flag = bool(getattr(model, "is_quantile", False))

    if out.dim() == 3:
        B, D1, D2 = out.shape
        # 1) 모양으로 1차 판별
        has_quant_axis = (D1 in (3, 5, 9)) or (D2 in (3, 5, 9))
        is_quantile = has_quant_axis or is_q_flag

        # 2) 최종 검증: 진짜 Q축이 없으면 분위수 경로 진입 금지
        if is_quantile and not has_quant_axis:
            # exo가 필요한 모델일 수 있으니 한번 더 시도
            try:
                Hm = _infer_horizon(model, default=120)
                exo = None
                if future_exo_cb is not None:
                    ex = future_exo_cb(0, Hm, device=device)     # (Hm,D)
                    exo = ex.unsqueeze(0).expand(x1.size(0), -1, -1)
                out2 = _safe_forward(model, x1.to(device), future_exo=exo)
                if isinstance(out2, (tuple, list)):
                    out2 = next(t for t in out2 if torch.is_tensor(t))
                if out2.dim() == 3:
                    has_quant_axis = (out2.shape[1] in (3,5,9)) or (out2.shape[2] in (3,5,9))
                else:
                    has_quant_axis = False
            except Exception:
                has_quant_axis = False

        if is_quantile and has_quant_axis:
            # IMS 롤아웃으로 120 스텝 분위수 생성
            q10, q50, q90 = _roll_quantile_ims(
                model, x1, horizon=120, device=device,
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

    # ---- 포인트형 모델: DMS→IMS로 120개월 ----
    f = DMSForecaster(
        model,
        target_channel=0,
        fill_mode="copy_last",
        lmm_mode="eval",
        predict_fn=None,
        ttm=None,
        future_exo_cb=future_exo_cb,
    )
    y120 = f.forecast_DMS_to_IMS(
        x_init=x1, horizon=120, device=device, extend="ims", context_policy="once",
    )
    return {"point": y120.squeeze(0).detach().cpu().numpy()}


# -------------------------
# Plot
# -------------------------
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

    # probe 유지(모양만 확인 용)
    probe_batch = next(iter(val_loader))
    xb_probe = probe_batch[0]
    if xb_probe.dim() == 2:
        xb_probe = xb_probe.unsqueeze(-1)
    _ = xb_probe[0:1].to(device)

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

            # GT
            y_true = None
            if use_truth and (yb is not None):
                y_true = yb[i:i+1].cpu().numpy().reshape(-1)

            # ---- 각 모델 120개월 예측 수집 ----
            preds_point = {}
            preds_q10, preds_q50, preds_q90 = {}, {}, {}

            for name, mdl in models.items():
                p = _predict_120_any(mdl, x1, device=device, future_exo_cb=future_exo_cb)
                preds_point[name] = p["point"]  # 항상 존재(길이=120 보장)

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

            # 1) 분위수 밴드 + 중앙선 (있는 모델만) — 이미 120 보장됨
            for nm in list(preds_q50.keys()):
                print(nm)
                q10 = preds_q10[nm]; q50 = preds_q50[nm]; q90 = preds_q90[nm]
                plt.fill_between(t_fut, q10, q90, alpha=0.15, label=f"{nm} P10–P90")
                plt.plot(t_fut, q50, linewidth=1.8, alpha=0.95, label=f"{nm} P50")

            # 2) 포인트 모델(혹은 분위수 없는)의 라인 — 이미 120 보장됨
            for nm, yhat in preds_point.items():
                if nm in preds_q50:
                    continue  # 중앙선과 중복 방지
                plt.plot(t_fut, yhat, label=nm, alpha=0.9)

            # 3) 최종 앙상블: “q90 기반” (길이 120 보장)
            stack_for_ens = []
            for nm in preds_point.keys():
                base = preds_q90[nm] if nm in preds_q90 else preds_point[nm]
                stack_for_ens.append(base)
            if stack_for_ens:
                M = np.vstack(stack_for_ens)              # (num_models, 120)
                ens_q90_mean = np.nanmean(M, axis=0)      # 시간별 평균
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
