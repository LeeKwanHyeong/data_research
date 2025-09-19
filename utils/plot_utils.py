import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import polars as pl
import matplotlib.dates as mdates
from utils.date_util import DateUtil


def _finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]

def plot_ims_samples_same_ylim(
    x_batch: torch.Tensor,
    y_hat: torch.Tensor,
    y_true: torch.Tensor | None = None,
    parts=None,
    k: int = 3,
    target_channel: int = 0,
    outdir: str = "./plots",
    prefix: str = "ims_pred",
    use_percentile_limits: bool = True,  # True면 p1~p99로 축 결정(이상치 영향↓)
):
    os.makedirs(outdir, exist_ok=True)

    x_cpu = x_batch.detach().cpu()
    if x_cpu.dim() == 2:
        x_cpu = x_cpu.unsqueeze(-1)  # [B,L] -> [B,L,1]
    B, L, C = x_cpu.shape

    yhat_cpu = y_hat.detach().cpu()
    H_pred = yhat_cpu.shape[1]

    ytrue_cpu = None
    H_true = None
    if y_true is not None:
        ytrue_cpu = y_true.detach().cpu()
        H_true = ytrue_cpu.shape[1]

    # ---- 공통 y축 범위 계산 (선택적으로 p1~p99 사용) ----
    vals = []
    for i in range(min(B, k)):
        vals.append(_finite(x_cpu[i, :, target_channel].numpy()))
        vals.append(_finite(yhat_cpu[i, :].numpy()))
        if ytrue_cpu is not None:
            vals.append(_finite(ytrue_cpu[i, :min(H_true, H_pred)].numpy()))
    all_vals = np.concatenate([v for v in vals if v.size > 0]) if vals else np.array([0.0])

    if use_percentile_limits and all_vals.size > 0:
        y_lo = np.percentile(all_vals, 1)
        y_hi = np.percentile(all_vals, 99)
        if y_lo == y_hi:  # 완전 평탄 보호
            y_lo -= 1.0
            y_hi += 1.0
    else:
        y_lo = np.nanmin(all_vals) if all_vals.size > 0 else -1.0
        y_hi = np.nanmax(all_vals) if all_vals.size > 0 else 1.0
        if not np.isfinite(y_lo): y_lo = -1.0
        if not np.isfinite(y_hi): y_hi = 1.0
        if y_lo == y_hi:
            y_lo -= 1.0
            y_hi += 1.0

    # ---- 샘플별 그래프 ----
    num = min(B, k)
    for i in range(num):
        hist = x_cpu[i, :, target_channel].numpy()
        pred = yhat_cpu[i].numpy()
        gt   = ytrue_cpu[i].numpy() if ytrue_cpu is not None else None

        t_hist = np.arange(L)
        t_pred = np.arange(L, L + H_pred)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_hist, hist, label="history")
        ax.plot(t_pred, pred, label=f"prediction(+{H_pred})")
        if gt is not None:
            ax.plot(t_pred[:min(H_pred, H_true)], gt[:min(H_pred, H_true)], label="ground truth")

        # 공통 y축 적용
        ax.set_ylim(y_lo, y_hi)

        # 샘플별 요약 통계(다른 값임을 제목에서 확인)
        title = f"Sample {i}"
        if parts is not None and len(parts) > i:
            title += f" | part={parts[i]}"
        title += f" | pred[min/med/max]={np.nanmin(pred):.2g}/{np.nanmedian(pred):.2g}/{np.nanmax(pred):.2g}"
        ax.set_title(title)
        ax.set_xlabel("time"); ax.set_ylabel("value")
        ax.legend(); ax.grid(True)
        fig.tight_layout()

        fn = os.path.join(outdir, f"{prefix}_sample{i}.png")
        fig.savefig(fn, dpi=150); plt.show(); plt.close(fig)
        print(f"Saved: {fn}")

def overlay_predictions(
    y_hat: torch.Tensor,
    parts=None,
    idxs: list[int] | None = None,
    outpath: str = "./plots/ims_overlay.png",
    title: str = "Overlay of predictions"
):
    """
    여러 샘플의 예측을 한 그래프에 겹쳐 비교(한 플롯).
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    y = y_hat.detach().cpu().numpy()
    B, H = y.shape
    if idxs is None:
        idxs = list(range(min(5, B)))  # 기본 5개

    fig, ax = plt.subplots(figsize=(10, 4))
    for i in idxs:
        lbl = f"idx {i}" + (f" ({parts[i]})" if parts is not None and len(parts) > i else "")
        ax.plot(np.arange(H), y[i], label=lbl)
    ax.set_title(title); ax.set_xlabel("horizon"); ax.set_ylabel("value")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.show(); plt.close(fig)
    print(f"Saved: {outpath}")



def plot_anchored_forecasts_yyyymm(
    df: pl.DataFrame,
    parts: list[str],
    y_hat: torch.Tensor,            # [N, H]
    plan_yyyymm: int,
    lookback: int,
    part_col: str = "oper_part_no",
    date_col: str = "demand_dt",
    qty_col: str = "demand_qty",
    k: int = 6,
    include_anchor: bool = False,   # False → plan 다음달부터 120개월
    outdir: str = "./plots",
    prefix: str = "anchored_yyyymm_120"
):
    """
    - parts 순서와 y_hat 행 순서가 일치한다고 가정
    - include_anchor=False: x축 미래 달력은 plan+1 ~ plan+120
      include_anchor=True : plan ~ plan+119
    """
    os.makedirs(outdir, exist_ok=True)

    pdf = df.to_pandas()
    preds = y_hat.detach().cpu().numpy()
    H = preds.shape[1]

    # 입력 히스토리 달력(앵커 직전 L개월) & 미래 달력
    hist_months = DateUtil.month_seq_ending_before(plan_yyyymm, lookback)  # [L]
    fut_months = DateUtil.next_n_months_from(plan_yyyymm, H, include_anchor=include_anchor)  # [H]

    # 파트별 month->value 맵 (빠른 조회)
    mp_group = {}
    for part in set(parts[:k]):
        sdf = pdf[pdf[part_col] == part].sort_values(date_col)
        months = sdf[date_col].astype(np.int64).to_numpy()
        vals = sdf[qty_col].astype(float).to_numpy()
        mp_group[part] = {int(m): float(v) for m, v in zip(months, vals)}

    for i, part in enumerate(parts[:k]):
        mp = mp_group.get(part, {})
        sdf = pdf[pdf[part_col] == part].sort_values(date_col)

        # 전체 실측 (정수 YYYYMM → datetime)
        all_m_int = sdf[date_col].astype(np.int64).to_numpy()
        all_v = sdf[qty_col].astype(float).to_numpy()
        all_m_dt = DateUtil.yyyymm_to_datetime(all_m_int)

        # 히스토리 (정수 → datetime)
        hist_vals = np.array([mp.get(int(m), np.nan) for m in hist_months], dtype=float)
        hist_dt = DateUtil.yyyymm_to_datetime(hist_months)

        # 예측 (정수 → datetime)
        pred = preds[i]
        fut_dt = DateUtil.yyyymm_to_datetime(fut_months)

        # 플롯 (datetime 축)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(all_m_dt, all_v, linewidth=3.0, alpha=0.35, label="actual (full)", zorder=1)
        ax.plot(hist_dt, hist_vals, linewidth=2.0, label=f"history (L={lookback})", zorder=2)
        ax.plot(fut_dt, pred, linewidth=2.0, label=f"forecast (+{H})", zorder=3)

        ax.set_title(f"oper_part_no={part} | plan={plan_yyyymm} | include_anchor={include_anchor}")
        ax.set_xlabel("time")
        ax.set_ylabel(qty_col)
        ax.grid(True)
        ax.legend()

        # 날짜 축 포맷(월 단위 눈금)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 3개월 간격 눈금
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
        fig.tight_layout()
        plt.show()


def plot_two_preds_same_ylim(
    x_batch: torch.Tensor,
    y_hat_a: torch.Tensor,           # (B, H) 예: preds_anchor_ims
    y_hat_b: torch.Tensor,           # (B, H) 예: preds_anchor_dms
    y_true: torch.Tensor | None = None,
    parts=None,
    k: int = 3,
    target_channel: int = 0,
    outdir: str = "./plots",
    prefix: str = "ims_vs_dms",
    label_a: str = "IMS",
    label_b: str = "DMS",
    use_percentile_limits: bool = True,
):
    os.makedirs(outdir, exist_ok=True)

    x_cpu = x_batch.detach().cpu()
    if x_cpu.dim() == 2:
        x_cpu = x_cpu.unsqueeze(-1)  # [B,L] -> [B,L,1]
    B, L, C = x_cpu.shape

    ya = y_hat_a.detach().cpu(); yb = y_hat_b.detach().cpu()
    H = ya.shape[1]
    assert yb.shape[1] == H, "두 예측의 horizon 길이가 달라요."

    yt = y_true.detach().cpu() if (y_true is not None) else None
    Ht = yt.shape[1] if yt is not None else None

    # 공통 y축
    vals = []
    for i in range(min(B, k)):
        vals.append(_finite(x_cpu[i, :, target_channel].numpy()))
        vals.append(_finite(ya[i, :].numpy()))
        vals.append(_finite(yb[i, :].numpy()))
        if yt is not None:
            vals.append(_finite(yt[i, :min(Ht, H)].numpy()))
    all_vals = np.concatenate([v for v in vals if v.size > 0]) if vals else np.array([0.0])

    if use_percentile_limits and all_vals.size > 0:
        y_lo = np.percentile(all_vals, 1); y_hi = np.percentile(all_vals, 99)
        if y_lo == y_hi: y_lo -= 1.0; y_hi += 1.0
    else:
        y_lo = np.nanmin(all_vals) if all_vals.size > 0 else -1.0
        y_hi = np.nanmax(all_vals) if all_vals.size > 0 else 1.0
        if not np.isfinite(y_lo): y_lo = -1.0
        if not np.isfinite(y_hi): y_hi = 1.0
        if y_lo == y_hi: y_lo -= 1.0; y_hi += 1.0

    # 샘플 플롯
    num = min(B, k)
    for i in range(num):
        hist = x_cpu[i, :, target_channel].numpy()
        a = ya[i].numpy()
        b = yb[i].numpy()
        gt = yt[i].numpy() if yt is not None else None

        t_hist = np.arange(L)
        t_pred = np.arange(L, L + H)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_hist, hist, label="history")
        ax.plot(t_pred, a, label=f"{label_a}(+{H})")
        ax.plot(t_pred, b, label=f"{label_b}(+{H})")
        if gt is not None:
            ax.plot(t_pred[:min(H, Ht)], gt[:min(H, Ht)], label="ground truth")

        ax.set_ylim(y_lo, y_hi)
        title = f"Sample {i}"
        if parts is not None and len(parts) > i:
            title += f" | part={parts[i]}"
        ax.set_title(title); ax.set_xlabel("time"); ax.set_ylabel("value")
        ax.legend(); ax.grid(True)
        fig.tight_layout()

        # fn = os.path.join(outdir, f"{prefix}_sample{i}.png")
        # fig.savefig(fn, dpi=150); plt.show(); plt.close(fig)
        # print(f"Saved: {fn}")

def plot_anchored_forecasts_yyyymm_multi(
    df: pl.DataFrame,
    parts: list[str],
    preds_dict: dict[str, torch.Tensor],  # {"IMS": (N,H), "DMS": (N,H)}
    plan_yyyymm: int,
    lookback: int,
    part_col: str = "oper_part_no",
    date_col: str = "demand_dt",
    qty_col: str = "demand_qty",
    k: int = 6,
    include_anchor: bool = False,
    outdir: str = "./plots",
    prefix: str = "anchored_compare"
):
    os.makedirs(outdir, exist_ok=True)
    pdf = df.to_pandas()

    # 텐서 → numpy
    preds_np = {lbl: t.detach().cpu().numpy() for lbl, t in preds_dict.items()}
    # 공통 H 확인
    Hs = {lbl: arr.shape[1] for lbl, arr in preds_np.items()}
    H = list(Hs.values())[0]
    assert all(h == H for h in Hs.values()), f"H mis-match {Hs}"

    # 달력
    hist_months = DateUtil.month_seq_ending_before(plan_yyyymm, lookback)
    fut_months  = DateUtil.next_n_months_from(plan_yyyymm, H, include_anchor=include_anchor)

    # 파트별 전체 시계열 캐시
    mp_group = {}
    for part in set(parts[:k]):
        sdf = pdf[pdf[part_col] == part].sort_values(date_col)
        months = sdf[date_col].astype(np.int64).to_numpy()
        vals = sdf[qty_col].astype(float).to_numpy()
        mp_group[part] = {int(m): float(v) for m, v in zip(months, vals)}

    for i, part in enumerate(parts[:k]):
        mp = mp_group.get(part, {})
        sdf = pdf[pdf[part_col] == part].sort_values(date_col)

        all_m_int = sdf[date_col].astype(np.int64).to_numpy()
        all_v     = sdf[qty_col].astype(float).to_numpy()
        all_m_dt  = DateUtil.yyyymm_to_datetime(all_m_int)

        hist_vals = np.array([mp.get(int(m), np.nan) for m in hist_months], dtype=float)
        hist_dt   = DateUtil.yyyymm_to_datetime(hist_months)
        fut_dt    = DateUtil.yyyymm_to_datetime(fut_months)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(all_m_dt, all_v, linewidth=3.0, alpha=0.35, label="actual (full)", zorder=1)
        ax.plot(hist_dt, hist_vals, linewidth=2.0, label=f"history (L={lookback})", zorder=2)

        # 여러 예측 라인
        for lbl, arr in preds_np.items():
            pred = arr[i]
            ax.plot(fut_dt, pred, linewidth=2.0, label=f"{lbl} (+{H})", zorder=3)

        ax.set_title(f"oper_part_no={part} | plan={plan_yyyymm} | include_anchor={include_anchor}")
        ax.set_xlabel("time"); ax.set_ylabel(qty_col)
        ax.grid(True); ax.legend()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for lbl in ax.get_xticklabels(): lbl.set_rotation(45)
        fig.tight_layout()
        # fn = os.path.join(outdir, f"{prefix}_{part}.png")
        # fig.savefig(fn, dpi=150); plt.show(); plt.close(fig)
        # print(f"Saved: {fn}")

def plot_anchored_forecasts_yyyyww_multi(
    df: pl.DataFrame,
    parts: list[str],
    preds_dict: dict[str, torch.Tensor],  # {"IMS": (N,H), "DMS": (N,H), ...}
    plan_yyyyww: int,
    lookback: int,
    part_col: str = "oper_part_no",
    date_col: str = "demand_dt",     # 주차 컬럼 (정수 YYYYWW)
    qty_col: str = "demand_qty",
    k: int = 6,
    include_anchor: bool = False,     # False → plan+1주부터 H주, True → plan주 포함
    outdir: str = "./plots",
    prefix: str = "anchored_compare_weekly",
    save: bool = True
):
    os.makedirs(outdir, exist_ok=True)
    pdf = df.to_pandas()

    # 텐서 → numpy
    preds_np = {lbl: t.detach().cpu().numpy() for lbl, t in preds_dict.items()}

    # 공통 H 체크
    Hs = {lbl: arr.shape[1] for lbl, arr in preds_np.items()}
    H = list(Hs.values())[0]
    assert all(h == H for h in Hs.values()), f"H mis-match {Hs}"

    # 주차 달력
    hist_weeks = DateUtil.week_seq_ending_before(plan_yyyyww, lookback)            # 길이 L
    fut_weeks  = DateUtil.next_n_weeks_from(plan_yyyyww, H, include_anchor=include_anchor)

    # 파트별 전체 시계열 캐시
    mp_group = {}
    for part in set(parts[:k]):
        sdf = pdf[pdf[part_col] == part].sort_values(date_col)
        weeks = sdf[date_col].astype(np.int64).to_numpy()
        vals  = sdf[qty_col].astype(float).to_numpy()
        mp_group[part] = {int(w): float(v) for w, v in zip(weeks, vals)}

    for i, part in enumerate(parts[:k]):
        mp = mp_group.get(part, {})
        sdf = pdf[pdf[part_col] == part].sort_values(date_col)

        # 전체 실측 (YYYYWW → datetime)
        all_w_int = sdf[date_col].astype(np.int64).to_numpy()
        all_v     = sdf[qty_col].astype(float).to_numpy()
        all_w_dt  = DateUtil.yyyyww_to_datetime(all_w_int)

        # 히스토리 (YYYYWW → datetime)
        hist_vals = np.array([mp.get(int(w), np.nan) for w in hist_weeks], dtype=float)
        hist_dt   = DateUtil.yyyyww_to_datetime(hist_weeks)

        # 미래 (YYYYWW → datetime)
        fut_dt    = DateUtil.yyyyww_to_datetime(fut_weeks)

        # 플롯
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(all_w_dt,  all_v,     linewidth=3.0, alpha=0.35, label="actual (full)", zorder=1)
        ax.plot(hist_dt,   hist_vals, linewidth=2.0,  label=f"history (L={lookback})", zorder=2)

        # 여러 예측 라인
        for lbl, arr in preds_np.items():
            pred = arr[i]
            ax.plot(fut_dt, pred, linewidth=2.0, label=f"{lbl} (+{H}w)", zorder=3)

        ax.set_title(f"oper_part_no={part} | plan={plan_yyyyww} | include_anchor={include_anchor}")
        ax.set_xlabel("time"); ax.set_ylabel(qty_col)
        ax.grid(True); ax.legend()

        # 주차 데이터지만, x축은 월 단위 메이저 눈금이 보기 좋습니다.
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for lbl_tick in ax.get_xticklabels():
            lbl_tick.set_rotation(45)

        fig.tight_layout()

        # if save:
        #     fn = os.path.join(outdir, f"{prefix}_{part}.png")
        #     fig.savefig(fn, dpi=150)
        #     print(f"Saved: {fn}")

        plt.show()
        plt.close(fig)
