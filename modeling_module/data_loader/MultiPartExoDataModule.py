import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Callable, Optional, Sequence, Dict, Any

from modeling_module.utils.date_util import DateUtil


# -----------------------------
# 유틸
# -----------------------------
def _to_numpy(x):
    if isinstance(x, pl.Series):
        return x.to_numpy()
    return np.asarray(x)


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
    def build_from_series(series: pl.Series, sort: bool = True, add_unk: bool = True) -> "CategoryIndexer":
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
        idx = CategoryIndexer(mapping)
        if add_unk:
            # 0 reserved for unknown
            pass
        return idx

    def id_of(self, value: Any) -> int:
        return self.mapping.get(value, self.unk_id)

    def map_series(self, s: pl.Series) -> np.ndarray:
        return np.asarray([self.id_of(v) for v in s.to_list()], dtype=np.int64)

    def state_dict(self) -> Dict[str, Any]:
        return {"mapping": self.mapping, "unk_id": self.unk_id}

    @staticmethod
    def from_state(state: Dict[str, Any]) -> "CategoryIndexer":
        ci = CategoryIndexer(mapping=state["mapping"])
        ci.unk_id = state.get("unk_id", 0)
        return ci


'''모델 forward 예시:

for x, y, part_ids, future_exo_cont, past_exo_cont, past_exo_cat in train_loader:
    out = model(
        x,
        future_exo=future_exo_cont,     # [B,H,E_fut] (float32)
        past_exo_cont=past_exo_cont,    # [B,L,E_cont] (float32)
        past_exo_cat=past_exo_cat,      # [B,L,E_cat]  (long, 정수ID)
        part_ids=part_ids               # list[str]
    )
'''


# ============================================================
# 1) Training Dataset
# ============================================================
class MultiPartExoTrainingDataset(Dataset):
    """
    외생변수(연속/범주)를 함께 제공하는 슬라이딩 윈도우 학습 Dataset.

    입력 df 스키마(필수 열):
      - part_no | demand_dt(int: YYYYWW or YYYYMM) | demand_qty(float)

    추천 exo:
      - 연속형: sequence, age_w, in_warranty, weeks_to_warranty_end, cumsum_qty ...
      - 범주형: site_id(정수), corp_id(정수) 등  ← *원핫으로 저장하지 마세요*

    반환:
      - x: [L, 1] float32
      - y: [H]    float32
      - future_exo_cont: [H, E_fut] float32
      - past_exo_cont:   [L, E_cont] float32
      - past_exo_cat:    [L, E_cat]  long (정수 ID)
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
        past_exo_cont_cols: Optional[Sequence[str]] = ("sequence", "age_w", "in_warranty", "weeks_to_warranty_end", "cumsum_qty"),
        past_exo_cat_cols: Optional[Sequence[str]]  = ("site_id",),   # 정수 ID여야 함
        future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
        cat_indexers: Optional[Dict[str, CategoryIndexer]] = None,  # (선택) cat 컬럼별 indexer 주입
    ):
        assert lookback and horizon
        self.lookback = int(lookback)
        self.horizon  = int(horizon)
        self.part_col = part_col
        self.date_col = date_col
        self.qty_col  = qty_col

        # None/빈 리스트도 허용
        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols  = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.future_exo_cb = future_exo_cb
        self.date_indexer  = date_indexer or (lambda x: x)

        # 카테고리 indexer (옵션): 전달되면 사용, 없으면 df의 값이 이미 정수라고 가정
        self.cat_indexers = cat_indexers or {}

        self.samples = []  # list[dict]

        grouped = df.partition_by(part_col)
        for g in grouped:
            g = g.sort(date_col)
            part = g[part_col][0]

            y_all = _to_numpy(g[qty_col]).astype(float)           # [T]
            d_all = _to_numpy(g[date_col]).astype(np.int64)        # [T]
            T = len(y_all)
            if T < self.lookback + self.horizon:
                continue

            # ----- 연속형 past exo -----
            if self.past_exo_cont_cols:
                cont_list = []
                for col in self.past_exo_cont_cols:
                    if col not in g.columns:
                        print(f"[TrainingDataset] missing past_exo_cont col: {col} -> skip")
                        continue
                    cont_list.append(_to_numpy(g[col]).astype(float))

                if cont_list:
                    exo_cont_mat = np.stack(cont_list, axis=-1)  # [T, E_cont]
                else:
                    exo_cont_mat = np.zeros((T, 0), dtype=float)
            else:
                exo_cont_mat = np.zeros((T, 0), dtype=float)

            # ----- 범주형 past exo (정수 ID) -----
            if self.past_exo_cat_cols:
                cat_list = []
                for col in self.past_exo_cat_cols:
                    if col not in g.columns:
                        print(f"[TrainingDataset] missing past_exo_cat col: {col} -> skip")
                        continue
                    s = g[col]
                    if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                        cat_list.append(_to_numpy(s).astype(np.int64))
                    else:
                        if col not in self.cat_indexers:
                            raise TypeError(
                                f"[TrainingDataset] categorical '{col}' must be integer IDs "
                                f"or provide cat_indexers[{col}]"
                            )
                        cat_list.append(self.cat_indexers[col].map_series(s))

                if cat_list:
                    exo_cat_mat = np.stack(cat_list, axis=-1)  # [T, E_cat]
                else:
                    exo_cat_mat = np.zeros((T, 0), dtype=np.int64)
            else:
                exo_cat_mat = np.zeros((T, 0), dtype=np.int64)

            # ----- 윈도우 생성 -----
            for i in range(T - self.lookback - self.horizon + 1):
                x_win = y_all[i:i+self.lookback]
                y_win = y_all[i+self.lookback:i+self.lookback+self.horizon]

                # past exo (연속형)
                if exo_cont_mat.size:
                    p_cont = exo_cont_mat[i:i+self.lookback, :]
                else:
                    p_cont = np.zeros((self.lookback, 0), dtype=float)

                # past exo (범주형)
                if exo_cat_mat.size:
                    p_cat = exo_cat_mat[i:i+self.lookback, :]
                else:
                    p_cat = np.zeros((self.lookback, 0), dtype=np.int64)

                # future exo (연속형)
                last_dt   = int(d_all[i+self.lookback-1])
                start_idx = int(self.date_indexer(last_dt)) + 1
                if self.future_exo_cb is not None:
                    fe = self.future_exo_cb(start_idx, self.horizon, device="cpu")
                    fe = fe.detach().cpu().numpy() if isinstance(fe, torch.Tensor) else np.asarray(fe, dtype=float)
                    assert fe.shape[0] == self.horizon, f"future_exo_cb must return (H, E), got {fe.shape}"
                else:
                    fe = np.zeros((self.horizon, 0), dtype=float)

                self.samples.append(dict(
                    x=x_win, y=y_win,
                    past_exo_cont=p_cont, past_exo_cat=p_cat,
                    future_exo_cont=fe,
                    part_id=part
                ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x  = torch.tensor(s["x"], dtype=torch.float32).unsqueeze(-1)          # [L,1]
        y  = torch.tensor(s["y"], dtype=torch.float32)                        # [H]
        pe_cont = torch.tensor(s["past_exo_cont"], dtype=torch.float32)       # [L,E_cont] (E_cont=0 가능)
        pe_cat  = torch.tensor(s["past_exo_cat"],  dtype=torch.long)          # [L,E_cat]  (E_cat=0 가능)
        fe_cont = torch.tensor(s["future_exo_cont"], dtype=torch.float32)     # [H,E_fut]  (E_fut=0 가능)
        return x, y, s["part_id"], fe_cont, pe_cont, pe_cat


# ============================================================
# 2) Anchored Inference Datasets
# ============================================================
class MultiPartExoAnchoredInferenceByYYYYWW(Dataset):
    """
    주(YYYYWW) 앵커 추론 Dataset (+ 연속/범주 exo)
    반환:
      - x: [L,1], part_id: str, future_exo_cont: [H,E_fut], past_exo_cont: [L,E_cont], past_exo_cat: [L,E_cat]
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
        past_exo_cont_cols: Optional[Sequence[str]] = ("sequence","age_w","in_warranty","weeks_to_warranty_end","cumsum_qty"),
        past_exo_cat_cols: Optional[Sequence[str]]  = ("site_id",),
        fill_missing: str = "ffill",
        target_back_weeks: int = 104,
        future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
        cat_indexers: Optional[Dict[str, CategoryIndexer]] = None,  # 미지 값 → UNK(0)
    ):
        assert fill_missing in ("ffill","zero","nan")
        self.lookback = int(lookback)
        self.horizon  = int(horizon)
        self.plan_yyyyww = int(plan_yyyyww)
        self.part_col = part_col
        self.date_col = date_col
        self.qty_col  = qty_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols  = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.fill_missing = fill_missing
        self.target_back_weeks = int(target_back_weeks)
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or (lambda x: x)
        self.cat_indexers = cat_indexers or {}

        self.inputs, self.part_ids = [], []
        self.past_exo_conts, self.past_exo_cats = [], []
        self.future_exo_conts = []

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
            qty_map = {int(w): float(v) for w, v in zip(weeks, vals)}
            earliest = int(weeks.min())

            # 수요 x 채우기
            x = np.empty(self.lookback, dtype=float)
            for i, ww in enumerate(win_weeks):
                if ww in qty_map:
                    x[i] = qty_map[ww]
                else:
                    if self.fill_missing == "zero":
                        x[i] = 0.0
                    elif self.fill_missing == "nan":
                        x[i] = np.nan
                    else:
                        prev, found = ww, False
                        for _ in range(self.target_back_weeks):
                            prev = DateUtil.add_weeks_yyyyww(prev, -1)
                            if prev < earliest: break
                            if prev in qty_map:
                                x[i] = qty_map[prev]; found = True; break
                        if not found: x[i] = 0.0
            if self.fill_missing == "nan" and not np.any(np.isfinite(x)):
                continue

            # ----- 연속형 past exo -----
            pe_cont = []
            for col in self.past_exo_cont_cols:
                if col not in g.columns:
                    print(f"[AnchoredYYYYWW] missing past_exo_cont col: {col} -> skip")
                    continue
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
                pe_cont.append(e)
            pe_cont_mat = np.stack(pe_cont, axis=-1) if pe_cont else np.zeros((self.lookback, 0), dtype=float)

            # ----- 범주형 past exo (정수 ID/UNK0) -----
            pe_cat = []
            for col in self.past_exo_cat_cols:
                if col not in g.columns:
                    print(f"[AnchoredYYYYWW] missing past_exo_cat col: {col} -> skip")
                    continue
                s = g[col]
                if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                               pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    mp = {int(w): int(v) for w, v in zip(weeks, _to_numpy(s).astype(np.int64))}
                    unk = 0
                else:
                    if col not in self.cat_indexers:
                        raise TypeError(
                            f"[AnchoredYYYYWW] categorical '{col}' must be integer IDs "
                            f"or provide cat_indexers[{col}]"
                        )
                    idxr = self.cat_indexers[col]
                    mp = {int(w): int(idxr.id_of(v)) for w, v in zip(weeks, s.to_list())}
                    unk = idxr.unk_id
                e = np.empty(self.lookback, dtype=np.int64)
                for i, ww in enumerate(win_weeks):
                    if ww in mp:
                        e[i] = mp[ww]
                    else:
                        if self.fill_missing in ("zero", "nan"):
                            e[i] = unk  # UNK=0
                        else:
                            prev, found = ww, False
                            cur = ww
                            for _ in range(self.target_back_weeks):
                                cur = DateUtil.add_weeks_yyyyww(cur, -1)
                                if cur < earliest: break
                                if cur in mp:
                                    e[i] = mp[cur]; found = True; break
                            if not found: e[i] = unk
                pe_cat.append(e)
            pe_cat_mat = np.stack(pe_cat, axis=-1) if pe_cat else np.zeros((self.lookback, 0), dtype=np.int64)

            # future exo (연속형)
            last_hist = int(win_weeks[-1])
            start_idx = int(self.date_indexer(last_hist)) + 1
            if self.future_exo_cb is not None:
                fe = self.future_exo_cb(start_idx, self.horizon, device="cpu")
                fe = fe.detach().cpu().numpy() if isinstance(fe, torch.Tensor) else np.asarray(fe, dtype=float)
                assert fe.shape[0] == self.horizon
            else:
                fe = np.zeros((self.horizon, 0), dtype=float)

            self.inputs.append(x)
            self.past_exo_conts.append(pe_cont_mat)
            self.past_exo_cats.append(pe_cat_mat)
            self.future_exo_conts.append(fe)
            self.part_ids.append(part)

    def __len__(self): return len(self.inputs)

    def __getitem__(self, idx):
        x   = torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(-1)
        peC = torch.tensor(self.past_exo_conts[idx], dtype=torch.float32)
        peK = torch.tensor(self.past_exo_cats[idx], dtype=torch.long)
        feC = torch.tensor(self.future_exo_conts[idx], dtype=torch.float32)
        return x, self.part_ids[idx], feC, peC, peK


class MultiPartExoAnchoredInferenceByYYYYMM(Dataset):
    """
    월(YYYYMM) 앵커 추론 Dataset (+ 연속/범주 exo)
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
        past_exo_cont_cols: Optional[Sequence[str]] = ("sequence","age_w","in_warranty","weeks_to_warranty_end","cumsum_qty"),
        past_exo_cat_cols: Optional[Sequence[str]]  = ("site_id",),
        fill_missing: str = "ffill",
        target_back_months: int = 120,
        future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
        cat_indexers: Optional[Dict[str, CategoryIndexer]] = None,
    ):
        assert fill_missing in ("ffill","zero","nan")
        self.lookback = int(lookback)
        self.horizon  = int(horizon)
        self.plan_yyyymm = int(plan_yyyymm)
        self.part_col = part_col
        self.date_col = date_col
        self.qty_col  = qty_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols  = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.fill_missing = fill_missing
        self.target_back_months = int(target_back_months)
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or (lambda x: x)
        self.cat_indexers = cat_indexers or {}

        self.inputs, self.part_ids = [], []
        self.past_exo_conts, self.past_exo_cats = [], []
        self.future_exo_conts = []

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

            # x
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

            # 연속형 exo
            pe_cont = []
            for col in self.past_exo_cont_cols:
                if col not in g.columns:
                    print(f"[AnchoredYYYYMM] missing past_exo_cont col: {col} -> skip")
                    continue
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
                pe_cont.append(e)
            pe_cont_mat = np.stack(pe_cont, axis=-1) if pe_cont else np.zeros((self.lookback, 0), dtype=float)

            # 범주형 exo
            pe_cat = []
            for col in self.past_exo_cat_cols:
                if col not in g.columns:
                    print(f"[AnchoredYYYYMM] missing past_exo_cat col: {col} -> skip")
                    continue
                s = g[col]
                if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                               pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    vp = {int(m): int(v) for m, v in zip(months, _to_numpy(s).astype(np.int64))}
                    unk = 0
                else:
                    if col not in self.cat_indexers:
                        raise TypeError(
                            f"[AnchoredYYYYMM] categorical '{col}' must be integer IDs "
                            f"or provide cat_indexers[{col}]"
                        )
                    idxr = self.cat_indexers[col]
                    vp = {int(m): int(idxr.id_of(v)) for m, v in zip(months, s.to_list())}
                    unk = idxr.unk_id
                e = np.empty(self.lookback, dtype=np.int64)
                for i, mm in enumerate(win_months):
                    if mm in vp:
                        e[i] = vp[mm]
                    else:
                        if self.fill_missing in ("zero", "nan"):
                            e[i] = unk
                        else:
                            prev, found = mm, False
                            cur = mm
                            for _ in range(self.target_back_months):
                                cur = DateUtil.add_months_yyyymm(cur, -1)
                                if cur < earliest: break
                                if cur in vp:
                                    e[i] = vp[cur]; found = True; break
                            if not found: e[i] = unk
                pe_cat.append(e)
            pe_cat_mat = np.stack(pe_cat, axis=-1) if pe_cat else np.zeros((self.lookback, 0), dtype=np.int64)

            # future exo
            last_hist = int(win_months[-1])
            start_idx = int(self.date_indexer(last_hist)) + 1
            if self.future_exo_cb is not None:
                fe = self.future_exo_cb(start_idx, self.horizon, device="cpu")
                fe = fe.detach().cpu().numpy() if isinstance(fe, torch.Tensor) else np.asarray(fe, dtype=float)
                assert fe.shape[0] == self.horizon
            else:
                fe = np.zeros((self.horizon, 0), dtype=float)

            self.inputs.append(x)
            self.past_exo_conts.append(pe_cont_mat)
            self.past_exo_cats.append(pe_cat_mat)
            self.future_exo_conts.append(fe)
            self.part_ids.append(part)

    def __len__(self): return len(self.inputs)

    def __getitem__(self, idx):
        x   = torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(-1)
        peC = torch.tensor(self.past_exo_conts[idx], dtype=torch.float32)
        peK = torch.tensor(self.past_exo_cats[idx], dtype=torch.long)
        feC = torch.tensor(self.future_exo_conts[idx], dtype=torch.float32)
        return x, self.part_ids[idx], feC, peC, peK


# ============================================================
# 3) DataModule
# ============================================================
class MultiPartExoDataModule:
    """
    외생변수를 포함한 멀티파트 시계열 학습/추론 DataModule.
    - 범주형 exo는 정수 ID로만 전달(UNK=0). one-hot 저장 금지.
    - 모델에서 nn.Embedding으로 처리 권장.
    - past_exo_cont_cols / past_exo_cat_cols / future_exo_cb 가 없어도 (None/빈 리스트) 동작.
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
        part_col: str = "oper_part_no",
        date_col: str = "demand_dt",
        qty_col: str = "demand_qty",
        past_exo_cont_cols: Optional[Sequence[str]] = ("sequence","age_w","in_warranty","weeks_to_warranty_end","cumsum_qty"),
        past_exo_cat_cols: Optional[Sequence[str]]  = ("site_id",),  # 정수ID 컬럼명
        fill_missing: str = "ffill",
        target_back_weeks: int = 104,
        target_back_months: int = 120,
        future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
        # (선택) 특정 cat 컬럼이 문자열이라면, 여기서 ID 매핑을 생성해 하위 Dataset에 주입
        build_cat_indexer_from: Optional[Sequence[str]] = ("site_cd",),  # 예: raw 문자열 컬럼명
        cat_indexer_target_col: Optional[str] = "site_id",               # ex) site_cd → site_id
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

        # None/빈 리스트도 허용
        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols  = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.fill_missing = fill_missing
        self.target_back_weeks = int(target_back_weeks)
        self.target_back_months = int(target_back_months)

        self.future_exo_cb = future_exo_cb
        self.date_indexer  = date_indexer or (lambda x: x)

        self.cat_indexers: Dict[str, CategoryIndexer] = {}

        # ----- (옵션) 문자열 cat → 정수ID 매핑 생성 & df에 부착 -----
        if build_cat_indexer_from:
            for raw_col in build_cat_indexer_from:
                if raw_col in self.df.columns:
                    # 문자열 기준으로 indexer 생성
                    idxr = CategoryIndexer.build_from_series(self.df[raw_col])
                    self.cat_indexers[raw_col] = idxr

                    # 타깃 ID 컬럼명 결정 (site_cd -> site_id)
                    target_col = cat_indexer_target_col if cat_indexer_target_col else f"{raw_col}_id"

                    # 정수 ID 컬럼 생성
                    self.df = self.df.with_columns(
                        pl.Series(
                            name=target_col,
                            values=idxr.map_series(self.df[raw_col])
                        ).cast(pl.Int32)
                    )

                    # cat exo 목록에 없으면 자동 추가
                    if target_col not in self.past_exo_cat_cols:
                        self.past_exo_cat_cols.append(target_col)

        self.train_dataset = None
        self.val_dataset   = None

    def setup(self):
        # 학습 Dataset
        full_dataset = MultiPartExoTrainingDataset(
            self.df, self.lookback, self.horizon,
            part_col=self.part_col,
            date_col=self.date_col,
            qty_col=self.qty_col,
            past_exo_cont_cols=self.past_exo_cont_cols,
            past_exo_cat_cols=self.past_exo_cat_cols,
            future_exo_cb=self.future_exo_cb,
            date_indexer=self.date_indexer,
            cat_indexers=self.cat_indexers,   # 문자열 cat 처리용
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
                past_exo_cont_cols=self.past_exo_cont_cols,
                past_exo_cat_cols=self.past_exo_cat_cols,
                fill_missing=self.fill_missing,
                target_back_weeks=self.target_back_weeks,
                future_exo_cb=self.future_exo_cb,
                date_indexer=self.date_indexer,
                cat_indexers=self.cat_indexers,
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
                past_exo_cont_cols=self.past_exo_cont_cols,
                past_exo_cat_cols=self.past_exo_cat_cols,
                fill_missing=self.fill_missing,
                target_back_months=self.target_back_months,
                future_exo_cb=self.future_exo_cb,
                date_indexer=self.date_indexer,
                cat_indexers=self.cat_indexers,
            )

        if parts_filter is not None:
            # 필요시 df를 미리 필터링해서 DataModule을 다시 생성하는 방식을 권장합니다.
            pass

        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)