"""plot_utils.py

Plot-only utilities for forecast results.

This module is intentionally decoupled from forecasting/model execution.
It can plot from:
- In-memory `rows` (list[dict])
- A `polars.DataFrame` that stores the same rows
- A parquet file that stores the same rows

Minimum expected row schema (per model per sample):
    {
        "part_id": str,
        "sample_idx": int,
        "model": str,
        "horizon": int,
        "y_pred_point": list[float],
        # optional quantiles
        "y_pred_q50": list[float] | None,
        "y_pred_q10": list[float] | None,
        "y_pred_q90": list[float] | None,
        # optional context
        "hist": list[float] | None,
        "y_true": list[float] | None,
        "freq": str | None,
        "mode": str | None,
        "plan_dt": int | None,
    }

The plot rendering is based on `_plot_single_series`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# ==============================
# Core plotting primitive
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
    xlabel: str = "Time",
):
    """Plot one series (history + optional truth + multiple model predictions)."""

    def _fit_len(a: Optional[np.ndarray], H: int) -> Optional[np.ndarray]:
        if a is None:
            return None
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.size > H:
            return a[:H]
        if a.size < H:
            return np.concatenate([a, np.full(H - a.size, np.nan)], axis=0)
        return a

    hist = None if hist is None else np.asarray(hist, dtype=float).reshape(-1)
    y_true = _fit_len(y_true, horizon)

    t_hist = np.arange(-len(hist) + 1, 1) if (hist is not None and hist.size > 0) else None
    t_fut = np.arange(1, horizon + 1)

    # zoom window
    if zoom_future:
        zL = int(zoom_len or horizon)
        zL = max(1, min(zL, horizon))
        t_view = t_fut[:zL]

        # history window heuristic: show up to 2*zL past points
        if t_hist is not None and len(t_hist) > zL * 2:
            t_hist_view = t_hist[-zL * 2 :]
            hist_view = hist[-zL * 2 :]
        else:
            t_hist_view = t_hist
            hist_view = hist
    else:
        zL = horizon
        t_view = t_fut
        t_hist_view = t_hist
        hist_view = hist

    plt.figure(figsize=(12, 6))

    # 1) history
    if hist_view is not None and hist_view.size > 0 and t_hist_view is not None:
        plt.plot(t_hist_view, hist_view, label="History", linewidth=2, alpha=0.8)

    # 2) truth
    if y_true is not None:
        plt.plot(t_view, y_true[:zL], label="True", linewidth=2.5, alpha=0.8)

    # 3) quantile models (band + median)
    for nm in sorted(set(preds_q50.keys())):
        q10 = _fit_len(preds_q10.get(nm), horizon)
        q50 = _fit_len(preds_q50.get(nm), horizon)
        q90 = _fit_len(preds_q90.get(nm), horizon)

        # Require q50; if q10/q90 missing, fall back to median line only
        if q50 is None:
            continue

        if q10 is not None and q90 is not None:
            plt.fill_between(t_view, q10[:zL], q90[:zL], alpha=0.15, label=f"{nm} P10–P90")

        plt.plot(t_view, q50[:zL], linewidth=2, linestyle="--", label=f"{nm} P50")

    # 4) point-only models
    for nm, yhat in preds_point.items():
        if nm in preds_q50:
            # already drawn as quantile/median
            continue
        a = _fit_len(yhat, horizon)
        if a is None:
            continue
        plt.plot(t_view, a[:zL], label=nm, linewidth=2, alpha=0.9)

    # 5) simple ensemble (mean of point series; if a model has q50 use that as point)
    stack = []
    for nm in sorted(set(list(preds_point.keys()) + list(preds_q50.keys()))):
        base = preds_q50.get(nm)
        if base is None:
            base = preds_point.get(nm)
        base = _fit_len(base, horizon)
        if base is not None:
            stack.append(base)
    if stack:
        M = np.vstack(stack)
        ens_mean = np.nanmean(M, axis=0)
        plt.plot(t_view, ens_mean[:zL], linestyle=":", linewidth=3, label="Ensemble (Mean)")

    # cosmetics
    plt.axvline(0, linewidth=1.2, linestyle="-")
    plt.title(title, fontsize=13)
    plt.xlabel(xlabel)
    plt.ylabel("Value")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# ==============================
# Ingestion helpers
# ==============================

def _list_to_np(v: Any) -> Optional[np.ndarray]:
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.astype(float).reshape(-1)
    try:
        return np.asarray(v, dtype=float).reshape(-1)
    except Exception:
        return None


def _first_non_null(values: Iterable[Any]) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _coerce_rows(data: Any) -> List[Dict[str, Any]]:
    """Coerce an input container into `list[dict]`.

    Accepted inputs
    - list[dict]
    - polars.DataFrame (recommended)
    - pandas.DataFrame
    """
    if isinstance(data, list):
        return data  # type: ignore[return-value]

    if pl is not None:
        try:
            if isinstance(data, pl.DataFrame):  # type: ignore[arg-type]
                return data.to_dicts()  # type: ignore[return-value]
        except Exception:
            pass

    if pd is not None:
        try:
            if isinstance(data, pd.DataFrame):  # type: ignore[arg-type]
                return data.to_dict(orient="records")  # type: ignore[return-value]
        except Exception:
            pass

    raise TypeError(
        "Expected one of: list[dict], polars.DataFrame, pandas.DataFrame. "
        f"Got: {type(data)}"
    )


def _group_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        pid = str(r.get("part_id"))
        sidx = int(r.get("sample_idx", 0))
        key = (pid, sidx)
        grouped.setdefault(key, []).append(r)
    return grouped


# ==============================
# Public plotting API
# ==============================

def plot_from_rows(
    rows: Any,
    *,
    part_id: Optional[str] = None,
    sample_idx: Optional[int] = None,
    max_plots: int = 10,
    out_dir: Optional[str] = None,
    show: bool = True,
    zoom: Union[bool, int] = False,
    title_prefix: Optional[str] = None,
    xlabel: str = "Time",
):
    """Plot from `rows` (list[dict] OR polars.DataFrame OR pandas.DataFrame).

    Note
    - This is the direct consumer of the schema you described:
      a per-model/per-sample row with `y_pred_*` columns.
    """

    rows = _coerce_rows(rows)

    if part_id is not None:
        rows = [r for r in rows if str(r.get("part_id")) == str(part_id)]
    if sample_idx is not None:
        rows = [r for r in rows if int(r.get("sample_idx", -1)) == int(sample_idx)]

    if not rows:
        raise ValueError("No rows matched the given filters (part_id/sample_idx).")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    zoom_future = False
    zoom_len = None
    if isinstance(zoom, bool) and zoom:
        zoom_future = True
    elif isinstance(zoom, int) and zoom > 0:
        zoom_future = True
        zoom_len = int(zoom)

    grouped = _group_rows(rows)
    plotted = 0

    for (pid, sidx), g in grouped.items():
        if plotted >= max_plots:
            break

        horizon = int(_first_non_null([rr.get("horizon") for rr in g]) or 0)
        if horizon <= 0:
            raise ValueError(f"Invalid horizon for part_id={pid}, sample_idx={sidx}.")

        hist = _list_to_np(_first_non_null([rr.get("hist") for rr in g]))
        y_true = _list_to_np(_first_non_null([rr.get("y_true") for rr in g]))

        freq = _first_non_null([rr.get("freq") for rr in g])
        mode = _first_non_null([rr.get("mode") for rr in g])
        plan_dt = _first_non_null([rr.get("plan_dt") for rr in g])

        preds_point: Dict[str, np.ndarray] = {}
        preds_q10: Dict[str, np.ndarray] = {}
        preds_q50: Dict[str, np.ndarray] = {}
        preds_q90: Dict[str, np.ndarray] = {}

        for rr in g:
            name = str(rr.get("model"))
            pp = _list_to_np(rr.get("y_pred_point"))
            if pp is not None:
                preds_point[name] = pp

            q50 = _list_to_np(rr.get("y_pred_q50"))
            q10 = _list_to_np(rr.get("y_pred_q10"))
            q90 = _list_to_np(rr.get("y_pred_q90"))

            if q50 is not None:
                preds_q50[name] = q50
            if q10 is not None:
                preds_q10[name] = q10
            if q90 is not None:
                preds_q90[name] = q90

        meta = []
        if title_prefix:
            meta.append(str(title_prefix))
        if mode is not None:
            meta.append(f"Mode={mode}")
        if freq is not None:
            meta.append(f"Freq={freq}")
        meta.append(f"H={horizon}")
        if plan_dt is not None:
            meta.append(f"PlanDt={plan_dt}")
        meta.append(f"ID={pid}")
        meta.append(f"Sample={sidx}")
        title = " | ".join(meta)

        safe_pid = str(pid).replace("/", "_").replace("\\", "_")
        out_path = (
            os.path.join(out_dir, f"plot_{safe_pid}_idx{sidx}_H{horizon}.png")
            if out_dir
            else None
        )

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
            xlabel=xlabel,
        )

        plotted += 1


def plot_from_parquet(
    parquet_path: str,
    *,
    part_id: Optional[str] = None,
    sample_idx: Optional[int] = None,
    max_plots: int = 10,
    out_dir: Optional[str] = None,
    show: bool = True,
    zoom: Union[bool, int] = False,
    title_prefix: Optional[str] = None,
    xlabel: str = "Time",
):
    """Plot from a parquet file that stores the same schema as `plot_from_rows`.

    Notes
    - Requires `polars`.
    - The parquet should contain one row per (part_id, sample_idx, model).
    """
    if pl is None:
        raise ImportError("polars is required for plot_from_parquet (pip install polars).")

    df = pl.read_parquet(parquet_path)

    if part_id is not None:
        df = df.filter(pl.col("part_id") == str(part_id))
    if sample_idx is not None:
        df = df.filter(pl.col("sample_idx") == int(sample_idx))

    if df.is_empty():
        raise ValueError("No rows matched the given filters (part_id/sample_idx).")

    plot_from_rows(
        df,
        part_id=None,
        sample_idx=None,
        max_plots=max_plots,
        out_dir=out_dir,
        show=show,
        zoom=zoom,
        title_prefix=title_prefix,
        xlabel=xlabel,
    )
