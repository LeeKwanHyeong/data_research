import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Callable, Optional, Sequence, Dict, Any
from datetime import datetime, timedelta

# 기존 DateUtil이 있다면 사용하고, 없으면 내부 로직 사용을 위해 import는 유지
try:
    from modeling_module.utils.date_util import DateUtil
except ImportError:
    DateUtil = None


# -----------------------------
# 유틸
# -----------------------------
def _to_numpy(x):
    if isinstance(x, pl.Series):
        return x.to_numpy()
    return np.asarray(x)


# 날짜 계산 헬퍼 함수 (Daily/Hourly 지원)
def _add_time(dt_int: int, amount: int, freq: str) -> int:
    """정수형 날짜(YYYYMM, YYYYWW, YYYYMMDD, YYYYMMDDHH)에 시간을 더하거나 뺌"""
    s = str(dt_int)

    if freq == 'hourly':
        # YYYYMMDDHH
        fmt = "%Y%m%d%H"
        dt_obj = datetime.strptime(s, fmt)
        new_dt = dt_obj + timedelta(hours=amount)
        return int(new_dt.strftime(fmt))

    elif freq == 'daily':
        # YYYYMMDD
        fmt = "%Y%m%d"
        dt_obj = datetime.strptime(s, fmt)
        new_dt = dt_obj + timedelta(days=amount)
        return int(new_dt.strftime(fmt))

    elif freq == 'weekly':
        # YYYYWW (기존 DateUtil 사용 권장, 없으면 datetime으로 근사 처리 불가하므로 DateUtil 필수)
        if DateUtil:
            return DateUtil.add_weeks_yyyyww(dt_int, amount)
        else:
            raise ImportError("Weekly logic requires DateUtil module.")

    elif freq == 'monthly':
        # YYYYMM
        if DateUtil:
            return DateUtil.add_months_yyyymm(dt_int, amount)
        else:
            # DateUtil 없을 경우 간단 구현
            y = dt_int // 100
            m = dt_int % 100
            m += amount
            while m < 1:
                m += 12
                y -= 1
            while m > 12:
                m -= 12
                y += 1
            return y * 100 + m
    return dt_int


def _generate_time_seq(plan_dt: int, length: int, freq: str) -> np.ndarray:
    """plan_dt 직전의 length 길이만큼의 과거 시퀀스 생성"""
    seq = []
    # plan_dt 바로 전 시점부터 역산
    current = _add_time(plan_dt, -1, freq)
    for _ in range(length):
        seq.append(current)
        current = _add_time(current, -1, freq)
    return np.array(seq[::-1], dtype=np.int64)


class CategoryIndexer:
    """
    문자열/임의 카테고리를 일관된 정수 ID로 변환하는 헬퍼.
    """

    def __init__(self, mapping: Optional[Dict[Any, int]] = None):
        self.unk_id = 0
        self.mapping: Dict[Any, int] = mapping or {}

    @staticmethod
    def build_from_series(series: pl.Series, sort: bool = True) -> "CategoryIndexer":
        vals = series.drop_nulls().unique().to_list()
        if sort:
            try:
                vals = sorted(vals)
            except Exception:
                pass
        mapping = {}
        next_id = 1  # 1..K
        for v in vals:
            if v not in mapping:
                mapping[v] = next_id
                next_id += 1
        return CategoryIndexer(mapping)

    def id_of(self, value: Any) -> int:
        return self.mapping.get(value, self.unk_id)

    def map_series(self, s: pl.Series) -> np.ndarray:
        return np.asarray([self.id_of(v) for v in s.to_list()], dtype=np.int64)


# ============================================================
# 1) Training Dataset
# ============================================================
class MultiPartExoTrainingDataset(Dataset):
    """
    슬라이딩 윈도우 학습 Dataset. (Daily/Hourly 등 모든 빈도 공용)
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
            past_exo_cont_cols: Optional[Sequence[str]] = None,
            past_exo_cat_cols: Optional[Sequence[str]] = None,
            future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
            date_indexer: Optional[Callable[[int], int]] = None,
            cat_indexers: Optional[Dict[str, CategoryIndexer]] = None,
    ):
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.part_col = part_col
        self.date_col = date_col
        self.qty_col = qty_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or (lambda x: x)
        self.cat_indexers = cat_indexers or {}

        self.samples = []

        grouped = df.partition_by(part_col)
        for g in grouped:
            g = g.sort(date_col)
            part = g[part_col][0]

            y_all = _to_numpy(g[qty_col]).astype(float)
            d_all = _to_numpy(g[date_col]).astype(np.int64)
            T = len(y_all)
            if T < self.lookback + self.horizon:
                continue

            # ----- 연속형 past exo -----
            if self.past_exo_cont_cols:
                cont_list = []
                for col in self.past_exo_cont_cols:
                    if col not in g.columns:
                        continue
                    cont_list.append(_to_numpy(g[col]).astype(float))
                exo_cont_mat = np.stack(cont_list, axis=-1) if cont_list else np.zeros((T, 0), dtype=float)
            else:
                exo_cont_mat = np.zeros((T, 0), dtype=float)

            # ----- 범주형 past exo -----
            if self.past_exo_cat_cols:
                cat_list = []
                for col in self.past_exo_cat_cols:
                    if col not in g.columns:
                        continue
                    s = g[col]
                    # 이미 정수형이면 그대로, 아니면 매핑
                    if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                        cat_list.append(_to_numpy(s).astype(np.int64))
                    else:
                        if col not in self.cat_indexers:
                            # 매퍼가 없으면 0 처리 혹은 에러. 여기선 에러
                            raise TypeError(f"Categorical '{col}' needs a CategoryIndexer or integer IDs.")
                        cat_list.append(self.cat_indexers[col].map_series(s))
                exo_cat_mat = np.stack(cat_list, axis=-1) if cat_list else np.zeros((T, 0), dtype=np.int64)
            else:
                exo_cat_mat = np.zeros((T, 0), dtype=np.int64)

            # ----- 윈도우 생성 -----
            # (데이터가 정렬되어 있고 빈 시간이 없다고 가정)
            for i in range(T - self.lookback - self.horizon + 1):
                x_win = y_all[i:i + self.lookback]
                y_win = y_all[i + self.lookback:i + self.lookback + self.horizon]

                p_cont = exo_cont_mat[i:i + self.lookback, :] if exo_cont_mat.size else np.zeros((self.lookback, 0),
                                                                                                 dtype=float)
                p_cat = exo_cat_mat[i:i + self.lookback, :] if exo_cat_mat.size else np.zeros((self.lookback, 0),
                                                                                              dtype=np.int64)

                # Future Exo
                last_dt = int(d_all[i + self.lookback - 1])
                start_idx = int(self.date_indexer(last_dt)) + 1

                fe = np.zeros((self.horizon, 0), dtype=float)
                if self.future_exo_cb is not None:
                    res = self.future_exo_cb(start_idx, self.horizon, device="cpu")
                    fe = res.detach().cpu().numpy() if isinstance(res, torch.Tensor) else np.asarray(res, dtype=float)

                self.samples.append(dict(
                    x=x_win, y=y_win,
                    past_exo_cont=p_cont, past_exo_cat=p_cat,
                    future_exo_cont=fe,
                    part_id=part
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = torch.tensor(s["x"], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(s["y"], dtype=torch.float32)
        pe_cont = torch.tensor(s["past_exo_cont"], dtype=torch.float32)
        pe_cat = torch.tensor(s["past_exo_cat"], dtype=torch.long)
        fe_cont = torch.tensor(s["future_exo_cont"], dtype=torch.float32)
        return x, y, s["part_id"], fe_cont, pe_cont, pe_cat


# ============================================================
# 2) Inference Dataset (Unified for Monthly/Weekly/Daily/Hourly)
# ============================================================
class MultiPartExoAnchoredInferenceDataset(Dataset):
    """
    특정 시점(plan_dt)을 기준으로 과거 데이터를 조회하여 추론 입력을 만드는 Dataset.
    freq에 따라 날짜 계산 로직을 분기합니다.
    """

    def __init__(
            self,
            df: pl.DataFrame,
            lookback: int,
            horizon: int,
            plan_dt: int,
            freq: str,  # 'monthly', 'weekly', 'daily', 'hourly'
            *,
            part_col: str = "part_no",
            date_col: str = "demand_dt",
            qty_col: str = "demand_qty",
            past_exo_cont_cols: Optional[Sequence[str]] = None,
            past_exo_cat_cols: Optional[Sequence[str]] = None,
            fill_missing: str = "ffill",
            target_back_steps: int = 100,  # 결측치 채울 때 얼마나 뒤를 볼지
            future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
            date_indexer: Optional[Callable[[int], int]] = None,
            cat_indexers: Optional[Dict[str, CategoryIndexer]] = None,
    ):
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.plan_dt = int(plan_dt)
        self.freq = freq.lower()

        self.part_col = part_col
        self.date_col = date_col
        self.qty_col = qty_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.fill_missing = fill_missing
        self.target_back_steps = int(target_back_steps)
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or (lambda x: x)
        self.cat_indexers = cat_indexers or {}

        self.inputs, self.part_ids = [], []
        self.past_exo_conts, self.past_exo_cats = [], []
        self.future_exo_conts = []

        # freq에 맞는 과거 시점 리스트 생성 (Ex: 과거 27주, 과거 24시간 등)
        win_dates = _generate_time_seq(self.plan_dt, self.lookback, self.freq)

        grouped = df.partition_by(part_col)
        for g in grouped:
            part = g[part_col][0]

            # 파티션 데이터를 맵으로 변환 (검색 속도 향상)
            dts = _to_numpy(g[date_col]).astype(np.int64)
            vals = _to_numpy(g[qty_col]).astype(float)

            if len(dts) == 0: continue

            qty_map = {int(d): float(v) for d, v in zip(dts, vals)}
            earliest = int(dts.min())

            # 1. Main Input (x) 채우기
            x = np.empty(self.lookback, dtype=float)
            for i, curr_dt in enumerate(win_dates):
                if curr_dt in qty_map:
                    x[i] = qty_map[curr_dt]
                else:
                    # 결측 처리
                    if self.fill_missing == "zero":
                        x[i] = 0.0
                    elif self.fill_missing == "nan":
                        x[i] = np.nan
                    else:  # ffill
                        prev, found = curr_dt, False
                        for _ in range(self.target_back_steps):
                            prev = _add_time(prev, -1, self.freq)
                            if prev < earliest: break
                            if prev in qty_map:
                                x[i] = qty_map[prev];
                                found = True;
                                break
                        if not found: x[i] = 0.0  # 못 찾으면 0

            # nan fill일 때 전체가 nan이면 스킵
            if self.fill_missing == "nan" and not np.any(np.isfinite(x)):
                continue

            # 2. Continuous Past Exo
            pe_cont_list = []
            for col in self.past_exo_cont_cols:
                if col not in g.columns: continue
                val_map = {int(d): float(v) for d, v in zip(dts, _to_numpy(g[col]).astype(float))}

                e = np.empty(self.lookback, dtype=float)
                for i, curr_dt in enumerate(win_dates):
                    if curr_dt in val_map:
                        e[i] = val_map[curr_dt]
                    else:
                        # 결측 처리 (위와 동일 로직)
                        if self.fill_missing == "zero":
                            e[i] = 0.0
                        elif self.fill_missing == "nan":
                            e[i] = np.nan
                        else:
                            prev, found = curr_dt, False
                            for _ in range(self.target_back_steps):
                                prev = _add_time(prev, -1, self.freq)
                                if prev < earliest: break
                                if prev in val_map:
                                    e[i] = val_map[prev];
                                    found = True;
                                    break
                            if not found: e[i] = 0.0
                pe_cont_list.append(e)

            pe_cont_mat = np.stack(pe_cont_list, axis=-1) if pe_cont_list else np.zeros((self.lookback, 0), dtype=float)

            # 3. Categorical Past Exo
            pe_cat_list = []
            for col in self.past_exo_cat_cols:
                if col not in g.columns: continue
                s = g[col]
                # Indexing
                if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    vals_int = _to_numpy(s).astype(np.int64)
                    unk = 0
                else:
                    if col not in self.cat_indexers:
                        # inference 시점에는 에러 대신 0(UNK) 처리하거나 strict하게 갈 수 있음
                        unk = 0
                        vals_int = np.zeros(len(s), dtype=np.int64)
                    else:
                        idxr = self.cat_indexers[col]
                        vals_int = np.array([idxr.id_of(v) for v in s.to_list()], dtype=np.int64)
                        unk = idxr.unk_id

                val_map = {int(d): int(v) for d, v in zip(dts, vals_int)}

                e = np.empty(self.lookback, dtype=np.int64)
                for i, curr_dt in enumerate(win_dates):
                    if curr_dt in val_map:
                        e[i] = val_map[curr_dt]
                    else:
                        if self.fill_missing in ("zero", "nan"):
                            e[i] = unk
                        else:
                            prev, found = curr_dt, False
                            for _ in range(self.target_back_steps):
                                prev = _add_time(prev, -1, self.freq)
                                if prev < earliest: break
                                if prev in val_map:
                                    e[i] = val_map[prev];
                                    found = True;
                                    break
                            if not found: e[i] = unk
                pe_cat_list.append(e)

            pe_cat_mat = np.stack(pe_cat_list, axis=-1) if pe_cat_list else np.zeros((self.lookback, 0), dtype=np.int64)

            # 4. Future Exo
            last_hist = int(win_dates[-1])
            start_idx = int(self.date_indexer(last_hist)) + 1
            fe = np.zeros((self.horizon, 0), dtype=float)
            if self.future_exo_cb is not None:
                res = self.future_exo_cb(start_idx, self.horizon, device="cpu")
                fe = res.detach().cpu().numpy() if isinstance(res, torch.Tensor) else np.asarray(res, dtype=float)

            self.inputs.append(x)
            self.past_exo_conts.append(pe_cont_mat)
            self.past_exo_cats.append(pe_cat_mat)
            self.future_exo_conts.append(fe)
            self.part_ids.append(part)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(-1)
        peC = torch.tensor(self.past_exo_conts[idx], dtype=torch.float32)
        peK = torch.tensor(self.past_exo_cats[idx], dtype=torch.long)
        feC = torch.tensor(self.future_exo_conts[idx], dtype=torch.float32)
        return x, self.part_ids[idx], feC, peC, peK


# ============================================================
# 3) Main DataModule
# ============================================================
class MultiPartExoDataModule:
    """
    - freq: 'monthly', 'weekly', 'daily', 'hourly' 중 하나 선택
    - date_col 형식:
       monthly -> YYYYMM (202401)
       weekly  -> YYYYWW (202401)
       daily   -> YYYYMMDD (20240101)
       hourly  -> YYYYMMDDHH (2024010112)
    """

    def __init__(
            self,
            df: pl.DataFrame,
            lookback: int,
            horizon: int,
            *,
            freq: str = 'weekly',  # 변경됨: is_running -> freq
            batch_size: int = 32,
            val_ratio: float = 0.2,
            shuffle: bool = False,
            seed: int = 42,
            part_col: str = "unique_id",
            date_col: str = "date",
            qty_col: str = "HUFL",
            past_exo_cont_cols: Optional[Sequence[str]] = None,
            past_exo_cat_cols: Optional[Sequence[str]] = None,
            fill_missing: str = "ffill",
            future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
            date_indexer: Optional[Callable[[int], int]] = None,
            build_cat_indexer_from: Optional[Sequence[str]] = None,
            cat_indexer_target_col: Optional[str] = None,
    ):
        self.df = df
        self.lookback = int(lookback)
        self.horizon = int(horizon)

        # Frequency 설정
        valid_freqs = ('monthly', 'weekly', 'daily', 'hourly')
        if freq not in valid_freqs:
            raise ValueError(f"freq must be one of {valid_freqs}, got '{freq}'")
        self.freq = freq

        self.batch_size = int(batch_size)
        self.val_ratio = float(val_ratio)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        self.part_col = part_col
        self.date_col = date_col
        self.qty_col = qty_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.fill_missing = fill_missing
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or (lambda x: x)

        self.cat_indexers: Dict[str, CategoryIndexer] = {}

        # 문자열 카테고리 -> 정수 ID 매핑
        if build_cat_indexer_from:
            for raw_col in build_cat_indexer_from:
                if raw_col in self.df.columns:
                    idxr = CategoryIndexer.build_from_series(self.df[raw_col])
                    self.cat_indexers[raw_col] = idxr

                    target_col = cat_indexer_target_col if cat_indexer_target_col else f"{raw_col}_id"

                    self.df = self.df.with_columns(
                        pl.Series(
                            name=target_col,
                            values=idxr.map_series(self.df[raw_col])
                        ).cast(pl.Int32)
                    )

                    if target_col not in self.past_exo_cat_cols:
                        self.past_exo_cat_cols.append(target_col)

        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        # 학습 Dataset 생성
        # TrainingDataset은 시계열 빈도(freq)와 무관하게
        # (Lookback+Horizon) 길이의 연속된 윈도우만 있으면 되므로 공용 클래스 사용
        full_dataset = MultiPartExoTrainingDataset(
            self.df, self.lookback, self.horizon,
            part_col=self.part_col,
            date_col=self.date_col,
            qty_col=self.qty_col,
            past_exo_cont_cols=self.past_exo_cont_cols,
            past_exo_cat_cols=self.past_exo_cat_cols,
            future_exo_cb=self.future_exo_cb,
            date_indexer=self.date_indexer,
            cat_indexers=self.cat_indexers,
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

    def get_inference_loader_at_plan(self, plan_dt: int):
        """
        plan_dt: 추론 시점 (YYYYMM, YYYYWW, YYYYMMDD, YYYYMMDDHH)
        """
        ds = MultiPartExoAnchoredInferenceDataset(
            df=self.df,
            lookback=self.lookback,
            horizon=self.horizon,
            plan_dt=int(plan_dt),
            freq=self.freq,  # 'monthly', 'weekly', 'daily', 'hourly'
            part_col=self.part_col,
            date_col=self.date_col,
            qty_col=self.qty_col,
            past_exo_cont_cols=self.past_exo_cont_cols,
            past_exo_cat_cols=self.past_exo_cat_cols,
            fill_missing=self.fill_missing,
            future_exo_cb=self.future_exo_cb,
            date_indexer=self.date_indexer,
            cat_indexers=self.cat_indexers,
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)