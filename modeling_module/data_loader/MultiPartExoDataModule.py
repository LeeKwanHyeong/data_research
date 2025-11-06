# MultiPartExoDataModule.py

import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Callable, Optional, Sequence

from modeling_module.utils.date_util import DateUtil


def _to_numpy(x):
    if isinstance(x, pl.Series):
        return x.to_numpy()
    return np.asarray(x)

'''모델 forward 예시:

for x, y, part_ids, future_exo, past_exo in train_loader:
    out = model(x, future_exo=future_exo, past_exo=past_exo, part_ids=part_ids)'''


class MultiPartExoTrainingDataset(Dataset):
    """
    외생변수(past_exo / future_exo)를 함께 제공하는 슬라이딩 윈도우 학습 Dataset.

    입력 df 스키마(열 최소):
      - part_no | demand_dt(int: YYYYWW/ YYYYMM) | demand_qty(float)
      - sequence | age_w | in_warranty | weeks_to_warranty_end | cumsum_qty (past exo 후보)

    반환:
      - x: [L, 1]
      - y: [H]
      - future_exo: [H, E_fut]
      - past_exo: [L, E_past]
      - part_id: str
    """
    def __init__(
        self,
        df: pl.DataFrame,
        lookback: int,
        horizon: int,
        *,
        part_col: str = "part_no",
        date_col: str = "demand_dt",
        qty_col: str = "demand_qty",
        past_exo_cols: Sequence[str] = ("sequence","age_w","in_warranty","weeks_to_warranty_end","cumsum_qty"),
        future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
    ):
        assert lookback and horizon
        self.lookback = int(lookback)
        self.horizon  = int(horizon)
        self.part_col = part_col
        self.date_col = date_col
        self.qty_col  = qty_col
        self.past_exo_cols = list(past_exo_cols)
        self.future_exo_cb = future_exo_cb
        self.date_indexer  = date_indexer or (lambda x: x)  # (필요 시 외부에서 주/월 → 절대 index 변환 함수 주입)

        self.samples = []  # list[dict]

        grouped = df.partition_by(part_col)
        for g in grouped:
            g = g.sort(date_col)
            part = g[part_col][0]

            y_all = _to_numpy(g[qty_col]).astype(float)
            d_all = _to_numpy(g[date_col]).astype(np.int64)

            # past exo 테이블 추출 (없으면 빈 배열)
            exo_past_mat = None
            if self.past_exo_cols:
                exo_list = []
                for col in self.past_exo_cols:
                    if col not in g.columns:
                        raise KeyError(f"[TrainingDataset] missing past_exo col: {col}")
                    exo_list.append(_to_numpy(g[col]).astype(float))
                exo_past_mat = np.stack(exo_list, axis=-1)  # [T, E_past]

            T = len(y_all)
            if T < self.lookback + self.horizon:
                continue

            for i in range(T - self.lookback - self.horizon + 1):
                x_win = y_all[i:i+self.lookback]
                y_win = y_all[i+self.lookback:i+self.lookback+self.horizon]

                # past exo window
                if exo_past_mat is not None:
                    p_win = exo_past_mat[i:i+self.lookback, :]
                else:
                    p_win = np.zeros((self.lookback, 0), dtype=float)

                # future exo 생성 (시작 인덱스: 입력 마지막 시점 다음)
                last_dt   = int(d_all[i+self.lookback-1])
                start_idx = int(self.date_indexer(last_dt)) + 1

                if self.future_exo_cb is not None:
                    fe = self.future_exo_cb(start_idx, self.horizon, device="cpu")
                    fe = fe.detach().cpu().numpy() if isinstance(fe, torch.Tensor) else np.asarray(fe, dtype=float)
                    assert fe.shape[0] == self.horizon, f"future_exo_cb must return (H, E), got {fe.shape}"
                else:
                    fe = np.zeros((self.horizon, 0), dtype=float)

                self.samples.append(dict(
                    x=x_win, y=y_win, past_exo=p_win, future_exo=fe, part_id=part
                ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x  = torch.tensor(s["x"], dtype=torch.float32).unsqueeze(-1)         # [L,1]
        y  = torch.tensor(s["y"], dtype=torch.float32)                       # [H]
        pe = torch.tensor(s["past_exo"], dtype=torch.float32)                # [L,E_past]
        fe = torch.tensor(s["future_exo"], dtype=torch.float32)              # [H,E_fut]
        return x, y, s["part_id"], fe, pe


class MultiPartExoAnchoredInferenceByYYYYWW(Dataset):
    """
    주차(YYYYWW) 기준 앵커 추론용 Dataset (+ 외생변수 지원)

    반환:
      - x: [L,1]
      - part_id: str
      - future_exo: [H,E_fut]
      - past_exo: [L,E_past]
    """
    def __init__(
        self,
        df: pl.DataFrame,
        lookback: int,
        horizon: int,
        plan_yyyyww: int,
        *,
        part_col: str = "part_no",
        date_col: str = "demand_dt",
        qty_col: str = "demand_qty",
        past_exo_cols: Sequence[str] = ("sequence","age_w","in_warranty","weeks_to_warranty_end","cumsum_qty"),
        fill_missing: str = "ffill",
        target_back_weeks: int = 104,
        future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
    ):
        assert fill_missing in ("ffill","zero","nan")
        self.lookback = int(lookback)
        self.horizon  = int(horizon)
        self.plan_yyyyww = int(plan_yyyyww)
        self.part_col = part_col
        self.date_col = date_col
        self.qty_col  = qty_col
        self.past_exo_cols = list(past_exo_cols)
        self.fill_missing = fill_missing
        self.target_back_weeks = int(target_back_weeks)
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or (lambda x: x)

        self.inputs, self.part_ids = [], []
        self.past_exos, self.future_exos = [], []

        grouped = df.partition_by(part_col)
        for g in grouped:
            g = g.sort(date_col)
            part = g[part_col][0]

            weeks = _to_numpy(g[date_col]).astype(np.int64)
            vals  = _to_numpy(g[qty_col]).astype(float)
            if len(weeks) == 0:
                continue

            # 과거 L주 캘린더
            win_weeks = DateUtil.week_seq_ending_before(self.plan_yyyyww, self.lookback)  # [L]
            qty_map = {int(w): float(v) for w,v in zip(weeks, vals)}

            # 수요 x 채우기
            x = np.empty(self.lookback, dtype=float)
            earliest = int(weeks.min())
            for i, ww in enumerate(win_weeks):
                if ww in qty_map:
                    x[i] = qty_map[ww]
                else:
                    if self.fill_missing == "zero":
                        x[i] = 0.0
                    elif self.fill_missing == "nan":
                        x[i] = np.nan
                    else:
                        # ffill
                        prev, found = ww, False
                        for _ in range(self.target_back_weeks):
                            prev = DateUtil.add_weeks_yyyyww(prev, -1)
                            if prev < earliest: break
                            if prev in qty_map:
                                x[i] = qty_map[prev]; found = True; break
                        if not found: x[i] = 0.0

            if self.fill_missing == "nan" and not np.any(np.isfinite(x)):
                continue

            # past_exo 채우기 (각 컬럼 동일 캘린더로 정렬 & ffill 정책 공유)
            pe = []
            for col in self.past_exo_cols:
                if col not in g.columns:
                    raise KeyError(f"[AnchoredYYYYWW] missing past_exo col: {col}")
                mp = {int(w): float(v) for w, v in zip(weeks, _to_numpy(g[col]).astype(float))}
                e = np.empty(self.lookback, dtype=float)
                for i, ww in enumerate(win_weeks):
                    if ww in mp:
                        e[i] = mp[ww]
                    else:
                        if self.fill_missing == "zero":
                            e[i] = 0.0
                        elif self.fill_missing == "nan":
                            e[i] = np.nan
                        else:
                            prev, found = ww, False
                            for _ in range(self.target_back_weeks):
                                prev = DateUtil.add_weeks_yyyyww(prev, -1)
                                if prev < earliest: break
                                if prev in mp:
                                    e[i] = mp[prev]; found = True; break
                            if not found: e[i] = 0.0
                pe.append(e)
            pe_mat = np.stack(pe, axis=-1) if pe else np.zeros((self.lookback, 0), dtype=float)

            # future_exo
            last_hist = int(win_weeks[-1])
            start_idx = int(self.date_indexer(last_hist)) + 1
            if self.future_exo_cb is not None:
                fe = self.future_exo_cb(start_idx, self.horizon, device="cpu")
                fe = fe.detach().cpu().numpy() if isinstance(fe, torch.Tensor) else np.asarray(fe, dtype=float)
                assert fe.shape[0] == self.horizon
            else:
                fe = np.zeros((self.horizon, 0), dtype=float)

            self.inputs.append(x)
            self.past_exos.append(pe_mat)
            self.future_exos.append(fe)
            self.part_ids.append(part)

    def __len__(self): return len(self.inputs)

    def __getitem__(self, idx):
        x  = torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(-1)   # [L,1]
        pe = torch.tensor(self.past_exos[idx], dtype=torch.float32)              # [L,E_past]
        fe = torch.tensor(self.future_exos[idx], dtype=torch.float32)            # [H,E_fut]
        return x, self.part_ids[idx], fe, pe


class MultiPartExoAnchoredInferenceByYYYYMM(Dataset):
    """
    월(YYYYMM) 기준 앵커 추론용 Dataset (+ 외생변수 지원)
    """
    def __init__(
        self,
        df: pl.DataFrame,
        lookback: int,
        horizon: int,
        plan_yyyymm: int,
        *,
        part_col: str = "part_no",
        date_col: str = "demand_dt",
        qty_col: str = "demand_qty",
        past_exo_cols: Sequence[str] = ("sequence","age_w","in_warranty","weeks_to_warranty_end","cumsum_qty"),
        fill_missing: str = "ffill",
        target_back_months: int = 120,
        future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
    ):
        assert fill_missing in ("ffill","zero","nan")
        self.lookback = int(lookback)
        self.horizon  = int(horizon)
        self.plan_yyyymm = int(plan_yyyymm)
        self.part_col = part_col
        self.date_col = date_col
        self.qty_col  = qty_col
        self.past_exo_cols = list(past_exo_cols)
        self.fill_missing = fill_missing
        self.target_back_months = int(target_back_months)
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or (lambda x: x)

        self.inputs, self.part_ids = [], []
        self.past_exos, self.future_exos = [], []

        grouped = df.partition_by(part_col)
        for g in grouped:
            g = g.sort(date_col)
            part = g[part_col][0]

            months = _to_numpy(g[date_col]).astype(np.int64)
            vals   = _to_numpy(g[qty_col]).astype(float)
            if len(months) == 0:
                continue

            win_months = DateUtil.month_seq_ending_before(self.plan_yyyymm, self.lookback)
            mp = {int(m): float(v) for m, v in zip(months, vals)}
            earliest = int(months.min())

            x = np.empty(self.lookback, dtype=float)
            for i, mm in enumerate(win_months):
                if mm in mp:
                    x[i] = mp[mm]
                else:
                    if self.fill_missing == "zero":
                        x[i] = 0.0
                    elif self.fill_missing == "nan":
                        x[i] = np.nan
                    else:
                        prev, found = mm, False
                        for _ in range(self.target_back_months):
                            prev = DateUtil.add_months_yyyymm(prev, -1)
                            if prev < earliest: break
                            if prev in mp:
                                x[i] = mp[prev]; found = True; break
                        if not found: x[i] = 0.0
            if self.fill_missing == "nan" and not np.any(np.isfinite(x)):
                continue

            # past_exo
            pe = []
            for col in self.past_exo_cols:
                if col not in g.columns:
                    raise KeyError(f"[AnchoredYYYYMM] missing past_exo col: {col}")
                vp = {int(m): float(v) for m, v in zip(months, _to_numpy(g[col]).astype(float))}
                e = np.empty(self.lookback, dtype=float)
                for i, mm in enumerate(win_months):
                    if mm in vp:
                        e[i] = vp[mm]
                    else:
                        if self.fill_missing == "zero":
                            e[i] = 0.0
                        elif self.fill_missing == "nan":
                            e[i] = np.nan
                        else:
                            prev, found = mm, False
                            for _ in range(self.target_back_months):
                                prev = DateUtil.add_months_yyyymm(prev, -1)
                                if prev < earliest: break
                                if prev in vp:
                                    e[i] = vp[prev]; found = True; break
                            if not found: e[i] = 0.0
                pe.append(e)
            pe_mat = np.stack(pe, axis=-1) if pe else np.zeros((self.lookback, 0), dtype=float)

            # future_exo
            last_hist = int(win_months[-1])
            start_idx = int(self.date_indexer(last_hist)) + 1
            if self.future_exo_cb is not None:
                fe = self.future_exo_cb(start_idx, self.horizon, device="cpu")
                fe = fe.detach().cpu().numpy() if isinstance(fe, torch.Tensor) else np.asarray(fe, dtype=float)
                assert fe.shape[0] == self.horizon
            else:
                fe = np.zeros((self.horizon, 0), dtype=float)

            self.inputs.append(x)
            self.past_exos.append(pe_mat)
            self.future_exos.append(fe)
            self.part_ids.append(part)

    def __len__(self): return len(self.inputs)

    def __getitem__(self, idx):
        x  = torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(-1)   # [L,1]
        pe = torch.tensor(self.past_exos[idx], dtype=torch.float32)              # [L,E_past]
        fe = torch.tensor(self.future_exos[idx], dtype=torch.float32)            # [H,E_fut]
        return x, self.part_ids[idx], fe, pe


class MultiPartExoDataModule:
    """
    외생변수를 포함한 멀티파트 시계열 학습/추론 DataModule.

    - 학습/검증: MultiPartExoTrainingDataset 사용
    - 추론: 주/월 앵커 기반 Dataset 사용
    - future_exo_cb(start_idx, H, device) 를 통해 미래 외생을 생성
    """
    def __init__(
        self,
        df: pl.DataFrame,
        lookback: int,
        horizon: int,
        *,
        is_running: bool,  # True: 주(YYYYWW), False: 월(YYYYMM)
        batch_size: int = 64,
        val_ratio: float = 0.2,
        shuffle: bool = True,
        seed: int = 42,
        part_col: str = "part_no",
        date_col: str = "demand_dt",
        qty_col: str = "demand_qty",
        past_exo_cols: Sequence[str] = ("sequence","age_w","in_warranty","weeks_to_warranty_end","cumsum_qty"),
        fill_missing: str = "ffill",
        target_back_weeks: int = 104,
        target_back_months: int = 120,
        future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
    ):
        self.df = df
        self.lookback = int(lookback)
        self.horizon  = int(horizon)
        self.is_running = bool(is_running)
        self.batch_size = int(batch_size)
        self.val_ratio = float(val_ratio)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        self.part_col = part_col
        self.date_col = date_col
        self.qty_col  = qty_col
        self.past_exo_cols = list(past_exo_cols)
        self.fill_missing = fill_missing
        self.target_back_weeks = int(target_back_weeks)
        self.target_back_months = int(target_back_months)

        self.future_exo_cb = future_exo_cb
        self.date_indexer  = date_indexer or (lambda x: x)

        self.train_dataset = None
        self.val_dataset   = None

    def setup(self):
        full_dataset = MultiPartExoTrainingDataset(
            self.df, self.lookback, self.horizon,
            part_col=self.part_col, date_col=self.date_col, qty_col=self.qty_col,
            past_exo_cols=self.past_exo_cols,
            future_exo_cb=self.future_exo_cb,
            date_indexer=self.date_indexer,
        )
        total_len = len(full_dataset)
        val_len = int(total_len * self.val_ratio)
        train_len = max(0, total_len - val_len)
        gen = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_len, val_len], generator=gen)

    def get_train_loader(self):
        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True
        )

    def get_val_loader(self):
        if self.val_dataset is None:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

    def get_inference_loader_at_plan(self, plan_dt: int, parts_filter=None):
        """
        plan_dt: YYYYWW (주모드) 또는 YYYYMM (월모드)
        """
        if self.is_running:
            ds = MultiPartExoAnchoredInferenceByYYYYWW(
                df=self.df,
                lookback=self.lookback,
                horizon=self.horizon,
                plan_yyyyww=int(plan_dt),
                part_col=self.part_col,
                date_col=self.date_col,
                qty_col=self.qty_col,
                past_exo_cols=self.past_exo_cols,
                fill_missing=self.fill_missing,
                target_back_weeks=self.target_back_weeks,
                future_exo_cb=self.future_exo_cb,
                date_indexer=self.date_indexer,
            )
        else:
            ds = MultiPartExoAnchoredInferenceByYYYYMM(
                df=self.df,
                lookback=self.lookback,
                horizon=self.horizon,
                plan_yyyymm=int(plan_dt),
                part_col=self.part_col,
                date_col=self.date_col,
                qty_col=self.qty_col,
                past_exo_cols=self.past_exo_cols,
                fill_missing=self.fill_missing,
                target_back_months=self.target_back_months,
                future_exo_cb=self.future_exo_cb,
                date_indexer=self.date_indexer,
            )

        # parts_filter 적용 (필요 시 간단 필터링)
        if parts_filter is not None:
            # 간단히 파트 필터를 적용하려면 DataLoader 레벨에서 배치 필터 대신
            # Dataset을 생성할 때 df를 미리 필터링하는 방식을 권장합니다.
            pass

        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)
