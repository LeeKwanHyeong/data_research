
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
from typing import Callable, Optional, Sequence, Dict, Any, List, Tuple
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
        # YYYYWW
        if DateUtil:
            return DateUtil.add_weeks_yyyyww(dt_int, amount)
        raise ImportError("Weekly logic requires DateUtil module.")

    elif freq == 'monthly':
        # YYYYMM
        if DateUtil:
            return DateUtil.add_months_yyyymm(dt_int, amount)

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
    current = _add_time(plan_dt, -1, freq)
    for _ in range(length):
        seq.append(current)
        current = _add_time(current, -1, freq)
    return np.array(seq[::-1], dtype=np.int64)


class CategoryIndexer:
    """
    문자열/임의 카테고리를 일관된 정수 ID로 변환하는 헬퍼.
    - UNK(미등록) 토큰은 0으로 예약
    - known values는 1..K 순번
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
# 1) Training Dataset (index_map 기반)
# ============================================================
class MultiPartExoTrainingDataset(Dataset):
    """
    슬라이딩 윈도우 학습 Dataset. (Daily/Hourly/Weekly/Monthly 공용)

    핵심:
      - 샘플을 dict로 전부 적재하지 않고, part별 배열 + 전역 index_map으로 구성
      - split_mode='multi' (id 단위 split) 지원을 위해 id->indices 매핑 제공

    반환:
      x: [L,1] float32
      y: [H]   float32
      id: str  (기본 unique_id)
      future_exo_cont: [H,E_fut] float32
      past_exo_cont:   [L,E_cont] float32
      past_exo_cat:    [L,E_cat]  long
    """
    def __init__(
        self,
        df: pl.DataFrame,
        lookback: int,
        horizon: int,
        *,
        id_col: str = "unique_id",
        date_col: str = "date",
        qty_col: str = "y",
        past_exo_cont_cols: Optional[Sequence[str]] = None,
        past_exo_cat_cols: Optional[Sequence[str]] = None,
        future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
        cat_indexers: Optional[Dict[str, CategoryIndexer]] = None,
    ):
        self.lookback = int(lookback)
        self.horizon = int(horizon)

        self.id_col = id_col
        self.date_col = date_col
        self.qty_col = qty_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or (lambda x: x)
        self.cat_indexers = cat_indexers or {}

        # id별 raw arrays 저장
        # self.series[id] = dict(y, d, exo_cont, exo_cat)
        self.series: Dict[str, Dict[str, np.ndarray]] = {}

        # 전역 인덱스: (id, start_i)
        self.index_map: List[Tuple[str, int]] = []

        # id -> global indices (split_mode='multi' 용)
        self.id_to_indices: Dict[str, List[int]] = {}

        if self.id_col not in df.columns:
            raise KeyError(f"id_col='{self.id_col}' not found in df.columns")
        if self.date_col not in df.columns:
            raise KeyError(f"date_col='{self.date_col}' not found in df.columns")
        if self.qty_col not in df.columns:
            raise KeyError(f"qty_col='{self.qty_col}' not found in df.columns")

        for g in df.partition_by(self.id_col):
            g = g.sort(self.date_col)
            uid = str(g[self.id_col][0])

            y_all = _to_numpy(g[self.qty_col]).astype(np.float32)     # [T]
            d_all = _to_numpy(g[self.date_col]).astype(np.int64)      # [T]
            T = len(y_all)
            if T < self.lookback + self.horizon:
                continue

            # ----- 연속형 past exo -----
            cont_list = []
            for col in self.past_exo_cont_cols:
                if col in g.columns:
                    cont_list.append(_to_numpy(g[col]).astype(np.float32))
            exo_cont = np.stack(cont_list, axis=-1) if cont_list else np.zeros((T, 0), dtype=np.float32)

            # ----- 범주형 past exo (정수 ID) -----
            cat_list = []
            for col in self.past_exo_cat_cols:
                if col not in g.columns:
                    continue
                s = g[col]
                if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    cat_list.append(_to_numpy(s).astype(np.int64))
                else:
                    if col not in self.cat_indexers:
                        raise TypeError(f"Categorical '{col}' needs a CategoryIndexer or integer IDs.")
                    cat_list.append(self.cat_indexers[col].map_series(s))
            exo_cat = np.stack(cat_list, axis=-1) if cat_list else np.zeros((T, 0), dtype=np.int64)

            self.series[uid] = {"y": y_all, "d": d_all, "exo_cont": exo_cont, "exo_cat": exo_cat}

            # ----- window index 생성 -----
            n_windows = T - self.lookback - self.horizon + 1
            if n_windows <= 0:
                continue

            self.id_to_indices[uid] = []
            for i in range(n_windows):
                gidx = len(self.index_map)
                self.index_map.append((uid, i))
                self.id_to_indices[uid].append(gidx)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        uid, i = self.index_map[idx]
        pack = self.series[uid]

        y_all = pack["y"]
        d_all = pack["d"]
        exo_cont = pack["exo_cont"]
        exo_cat = pack["exo_cat"]

        L = self.lookback
        H = self.horizon

        x_win = y_all[i:i + L]                 # [L]
        y_win = y_all[i + L:i + L + H]         # [H]

        pe_cont = exo_cont[i:i + L, :] if exo_cont.shape[1] > 0 else np.zeros((L, 0), dtype=np.float32)
        pe_cat = exo_cat[i:i + L, :] if exo_cat.shape[1] > 0 else np.zeros((L, 0), dtype=np.int64)

        # Future Exo
        last_dt = int(d_all[i + L - 1])
        start_idx = int(self.date_indexer(last_dt)) + 1

        fe = np.zeros((H, 0), dtype=np.float32)
        if self.future_exo_cb is not None:
            res = self.future_exo_cb(start_idx, H, device="cpu")
            fe = res.detach().cpu().numpy() if isinstance(res, torch.Tensor) else np.asarray(res, dtype=np.float32)
            if fe.shape[0] != H:
                raise ValueError(f"future_exo_cb must return (H, E), got {fe.shape}")

        x = torch.tensor(x_win, dtype=torch.float32).unsqueeze(-1)  # [L,1]
        y = torch.tensor(y_win, dtype=torch.float32)                # [H]
        pe_cont_t = torch.tensor(pe_cont, dtype=torch.float32)      # [L,E_cont]
        pe_cat_t = torch.tensor(pe_cat, dtype=torch.long)           # [L,E_cat]
        fe_t = torch.tensor(fe, dtype=torch.float32)                # [H,E_fut]

        return x, y, uid, fe_t, pe_cont_t, pe_cat_t


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
            id_col: str = "unique_id",
            date_col: str = "date",
            qty_col: str = "y",
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

        self.id_col = id_col
        self.date_col = date_col
        self.qty_col = qty_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.fill_missing = fill_missing
        self.target_back_steps = int(target_back_steps)
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or (lambda x: x)
        self.cat_indexers = cat_indexers or {}

        self.inputs, self.ids = [], []
        self.past_exo_conts, self.past_exo_cats = [], []
        self.future_exo_conts = []

        # freq에 맞는 과거 시점 리스트 생성 (Ex: 과거 27주, 과거 24시간 등)
        win_dates = _generate_time_seq(self.plan_dt, self.lookback, self.freq)

        grouped = df.partition_by(self.id_col)
        for g in grouped:
            uid = str(g[self.id_col][0])

            dts = _to_numpy(g[self.date_col]).astype(np.int64)
            vals = _to_numpy(g[self.qty_col]).astype(float)
            if len(dts) == 0:
                continue

            qty_map = {int(d): float(v) for d, v in zip(dts, vals)}
            earliest = int(dts.min())

            # 1) x
            x = np.empty(self.lookback, dtype=float)
            for i, curr_dt in enumerate(win_dates):
                if curr_dt in qty_map:
                    x[i] = qty_map[curr_dt]
                else:
                    if self.fill_missing == "zero":
                        x[i] = 0.0
                    elif self.fill_missing == "nan":
                        x[i] = np.nan
                    else:
                        prev, found = curr_dt, False
                        for _ in range(self.target_back_steps):
                            prev = _add_time(prev, -1, self.freq)
                            if prev < earliest:
                                break
                            if prev in qty_map:
                                x[i] = qty_map[prev]
                                found = True
                                break
                        if not found:
                            x[i] = 0.0

            if self.fill_missing == "nan" and not np.any(np.isfinite(x)):
                continue

            # 2) Continuous Past Exo
            pe_cont_list = []
            for col in self.past_exo_cont_cols:
                if col not in g.columns:
                    continue
                val_map = {int(d): float(v) for d, v in zip(dts, _to_numpy(g[col]).astype(float))}

                e = np.empty(self.lookback, dtype=float)
                for i, curr_dt in enumerate(win_dates):
                    if curr_dt in val_map:
                        e[i] = val_map[curr_dt]
                    else:
                        if self.fill_missing == "zero":
                            e[i] = 0.0
                        elif self.fill_missing == "nan":
                            e[i] = np.nan
                        else:
                            prev, found = curr_dt, False
                            for _ in range(self.target_back_steps):
                                prev = _add_time(prev, -1, self.freq)
                                if prev < earliest:
                                    break
                                if prev in val_map:
                                    e[i] = val_map[prev]
                                    found = True
                                    break
                            if not found:
                                e[i] = 0.0
                pe_cont_list.append(e)

            pe_cont_mat = np.stack(pe_cont_list, axis=-1) if pe_cont_list else np.zeros((self.lookback, 0), dtype=float)

            # 3) Categorical Past Exo
            pe_cat_list = []
            for col in self.past_exo_cat_cols:
                if col not in g.columns:
                    continue
                s = g[col]

                if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    vals_int = _to_numpy(s).astype(np.int64)
                    unk = 0
                else:
                    if col not in self.cat_indexers:
                        # inference는 UNK(0) 처리
                        vals_int = np.zeros(len(s), dtype=np.int64)
                        unk = 0
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
                                if prev < earliest:
                                    break
                                if prev in val_map:
                                    e[i] = val_map[prev]
                                    found = True
                                    break
                            if not found:
                                e[i] = unk

                pe_cat_list.append(e)

            pe_cat_mat = np.stack(pe_cat_list, axis=-1) if pe_cat_list else np.zeros((self.lookback, 0), dtype=np.int64)

            # 4) Future Exo
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
            self.ids.append(uid)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(-1)
        peC = torch.tensor(self.past_exo_conts[idx], dtype=torch.float32)
        peK = torch.tensor(self.past_exo_cats[idx], dtype=torch.long)
        feC = torch.tensor(self.future_exo_conts[idx], dtype=torch.float32)
        return x, self.ids[idx], feC, peC, peK


# ============================================================
# 3) Main DataModule (split_mode: window | multi)
# ============================================================
class MultiPartExoDataModule:
    """
    - freq: 'monthly', 'weekly', 'daily', 'hourly' 중 하나 선택
    - date_col 형식:
       monthly -> YYYYMM (202401)
       weekly  -> YYYYWW (202401)   (DateUtil 필요)
       daily   -> YYYYMMDD (20240101)
       hourly  -> YYYYMMDDHH (2024010112)

    split_mode:
      - 'window': 전체 window 샘플을 랜덤 split (id가 train/val에 섞일 수 있음)
      - 'multi' : id(=id_col) 단위 split (train/val id disjoint, leakage 최소화)
    """

    def __init__(
            self,
            df: pl.DataFrame,
            lookback: int,
            horizon: int,
            *,
            freq: str = 'weekly',
            batch_size: int = 32,
            val_ratio: float = 0.2,
            shuffle: bool = True,          # 기본 True 권장 (동일 id 배치 문제 완화)
            seed: int = 42,
            id_col: str = "unique_id",      # 디폴트 unique_id (필요시 변경)
            date_col: str = "date",
            y_col: str = "HUFL",
            past_exo_cont_cols: Optional[Sequence[str]] = None,
            past_exo_cat_cols: Optional[Sequence[str]] = None,
            fill_missing: str = "ffill",
            target_back_steps: int = 100,
            future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
            date_indexer: Optional[Callable[[int], int]] = None,
            build_cat_indexer_from: Optional[Sequence[str]] = None,
            cat_indexer_target_col: Optional[str] = None,
            split_mode: str = "window",     # 'window' | 'multi'
    ):
        self.df = df
        self.lookback = int(lookback)
        self.horizon = int(horizon)

        valid_freqs = ('monthly', 'weekly', 'daily', 'hourly')
        if freq not in valid_freqs:
            raise ValueError(f"freq must be one of {valid_freqs}, got '{freq}'")
        self.freq = freq

        if split_mode not in ("window", "multi"):
            raise ValueError("split_mode must be 'window' or 'multi'")
        self.split_mode = split_mode

        self.batch_size = int(batch_size)
        self.val_ratio = float(val_ratio)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        self.id_col = id_col
        self.date_col = date_col
        self.qty_col = y_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.fill_missing = fill_missing
        self.target_back_steps = int(target_back_steps)
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or (lambda x: x)

        self.cat_indexers: Dict[str, CategoryIndexer] = {}

        # 문자열 카테고리 -> 정수 ID 매핑 (원본 컬럼명 기준 indexer 저장)
        if build_cat_indexer_from:
            for raw_col in build_cat_indexer_from:
                if raw_col in self.df.columns:
                    idxr = CategoryIndexer.build_from_series(self.df[raw_col])
                    self.cat_indexers[raw_col] = idxr

                    target_col = cat_indexer_target_col if cat_indexer_target_col else f"{raw_col}_id"

                    self.df = self.df.with_columns(
                        pl.Series(name=target_col, values=idxr.map_series(self.df[raw_col])).cast(pl.Int32)
                    )

                    if target_col not in self.past_exo_cat_cols:
                        self.past_exo_cat_cols.append(target_col)

        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        full_dataset = MultiPartExoTrainingDataset(
            self.df,
            self.lookback,
            self.horizon,
            id_col=self.id_col,
            date_col=self.date_col,
            qty_col=self.qty_col,
            past_exo_cont_cols=self.past_exo_cont_cols,
            past_exo_cat_cols=self.past_exo_cat_cols,
            future_exo_cb=self.future_exo_cb,
            date_indexer=self.date_indexer,
            cat_indexers=self.cat_indexers,
        )

        total_len = len(full_dataset)
        if total_len == 0:
            self.train_dataset = full_dataset
            self.val_dataset = full_dataset
            return

        gen = torch.Generator().manual_seed(self.seed)

        # -------------------------
        # 1) window split
        # -------------------------
        if self.split_mode == "window":
            val_len = int(total_len * self.val_ratio)
            train_len = max(0, total_len - val_len)
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_len, val_len], generator=gen)
            return

        # -------------------------
        # 2) multi split (id 단위)
        # -------------------------
        ids = list(full_dataset.id_to_indices.keys())
        if len(ids) <= 1:
            # id 1개면 multi split 불가 -> window split로 degrade
            val_len = int(total_len * self.val_ratio)
            train_len = max(0, total_len - val_len)
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_len, val_len], generator=gen)
            return

        # id별 window 수
        id_counts = {uid: len(full_dataset.id_to_indices[uid]) for uid in ids}
        total_windows = sum(id_counts.values())
        target_val_windows = int(total_windows * self.val_ratio)

        rng = np.random.default_rng(self.seed)
        rng.shuffle(ids)

        # val ids 선정 (window 수 기준 greedy)
        val_ids: List[str] = []
        cur = 0
        for uid in ids:
            if cur >= target_val_windows and len(val_ids) > 0:
                break
            val_ids.append(uid)
            cur += id_counts[uid]

        val_id_set = set(val_ids)
        train_ids = [uid for uid in ids if uid not in val_id_set]

        train_indices: List[int] = []
        for uid in train_ids:
            train_indices.extend(full_dataset.id_to_indices[uid])

        val_indices: List[int] = []
        for uid in val_ids:
            val_indices.extend(full_dataset.id_to_indices[uid])

        # 안정성: 비었으면 window split로 degrade
        if len(train_indices) == 0 or len(val_indices) == 0:
            val_len = int(total_len * self.val_ratio)
            train_len = max(0, total_len - val_len)
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_len, val_len], generator=gen)
            return

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)

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
            freq=self.freq,
            id_col=self.id_col,
            date_col=self.date_col,
            qty_col=self.qty_col,
            past_exo_cont_cols=self.past_exo_cont_cols,
            past_exo_cat_cols=self.past_exo_cat_cols,
            fill_missing=self.fill_missing,
            target_back_steps=self.target_back_steps,
            future_exo_cb=self.future_exo_cb,
            date_indexer=self.date_indexer,
            cat_indexers=self.cat_indexers,
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)
