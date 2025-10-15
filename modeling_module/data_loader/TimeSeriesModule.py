import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

from modeling_module.utils.date_util import DateUtil


class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback, horizon):
        '''
            data: 1D numpy array or list (time series)
            lookback: input sequence length
            horizon: prediction length
        '''
        self.data = data
        self.lookback = lookback
        self.horizon = horizon
        self.samples = []

        for i in range(len(data) - lookback - horizon):
            x_seq = data[i:i + lookback]
            y_seq = data[i + lookback: i + lookback + horizon]
            self.samples.append((x_seq, y_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, y_seq = self.samples[idx]
        return torch.tensor(x_seq, dtype = torch.float32).unsqueeze(-1), \
               torch.tensor(y_seq, dtype = torch.float32) # [lookback, 1], [horizon]



class MultiPartInferenceDataset(Dataset):
    def __init__(self, df: pl.DataFrame, config):
        self.lookback = config.lookback
        self.inputs = []
        self.part_ids = []

        grouped = df.partition_by('oper_part_no')
        for g in grouped:
            series = g.sort('demand_dt')['demand_qty'].to_numpy()
            part_no = g['oper_part_no'][0]

            if len(series) < self.lookback:
                continue

            x_seq = series[-self.lookback:]
            self.inputs.append(x_seq)
            self.part_ids.append(part_no)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(-1),  # [lookback, 1]
            self.part_ids[idx]  # string
        )

class MultiPartTrainingDataset(Dataset):
    def __init__(self, df: pl.DataFrame, config):
        self.lookback = config.lookback
        self.horizon = config.horizon

        self.samples = []
        self.part_ids = []


        grouped = df.partition_by('oper_part_no')
        for g in grouped:
            series = g['demand_qty'].to_numpy()
            part_no = g['oper_part_no'][0]
            if len(series) < self.lookback + self.horizon:
                continue
            for i in range(len(series) - self.lookback - self.horizon + 1):
                x_seq = series[i: i+self.lookback]
                y_seq = series[i+self.lookback: i+self.lookback+self.horizon]
                self.samples.append((x_seq, y_seq))
                self.part_ids.append(part_no)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, y_seq = self.samples[idx]
        return (
            torch.tensor(x_seq, dtype = torch.float32).unsqueeze(-1),
            torch.tensor(y_seq, dtype = torch.float32),
            self.part_ids[idx]
        )

class MultiPartAnchoredInferenceByYYYYWW(Dataset):
    """
    주차(YYYYWW, ISO week) 기준 앵커드 추론 입력 생성용 Dataset.

    df: pl.DataFrame({'oper_part_no': str, 'demand_wk': int(YYYYWW), 'demand_qty': float})
    plan_yyyyww: int (예: 202252). '그 이후부터' 예측하려면,
      - 입력은 plan_yyyyww '직전' L주
      - 첫 예측 주차는 보통 plan_yyyyww (포함) 또는 plan+1 (이후) → 플롯/추론단에서 결정
    fill_missing: 'ffill' | 'zero' | 'nan'
      - 입력 윈도우 내 존재하지 않는 주를 어떻게 채울지
    parts_filter: 특정 part만 추론하고 싶을 때 리스트/셋 전달
    target_horizon: ffill 백트래킹 시 최대로 뒤로 거슬러갈 주 수(보호용)
    """

    def __init__(self,
                 df: pl.DataFrame,
                 lookback: int,
                 plan_yyyyww: int,
                 part_col: str = 'oper_part_no',
                 date_col: str = 'demand_wk',     # 주차 컬럼명(정수 YYYYWW)
                 qty_col: str = 'demand_qty',
                 parts_filter=None,
                 fill_missing: str = 'ffill',
                 target_horizon: int = 104):       # 2년치 백업 탐색 기본
        self.lookback = int(lookback)
        self.plan_yyyyww = int(plan_yyyyww)
        self.part_col = part_col
        self.date_col = date_col
        self.qty_col = qty_col
        assert fill_missing in ('ffill', 'zero', 'nan')
        self.fill_missing = fill_missing
        self.target_horizon = int(target_horizon)

        self.inputs: list[np.ndarray] = []     # 각 파트 입력 시퀀스 [L]
        self.part_ids: list[str] = []          # part_no
        self.hist_yyyyww: list[list[int]] = [] # 각 파트의 윈도우 주차 리스트

        parts_set = set(parts_filter) if parts_filter is not None else None

        # 파트별 그룹
        grouped = df.partition_by(part_col)
        for g in grouped:
            part = g[part_col][0]
            if parts_set is not None and part not in parts_set:
                continue

            gd = g.select([date_col, qty_col]).sort(date_col)
            weeks = gd[date_col].to_numpy().astype(np.int64)   # YYYYWW (ISO week)
            values = gd[qty_col].to_numpy().astype(float)

            if len(weeks) == 0:
                continue

            # 입력 윈도우(앵커 직전 L주)
            win_weeks = DateUtil.week_seq_ending_before(self.plan_yyyyww, self.lookback)  # 길이 L
            mp = {int(w): float(v) for w, v in zip(weeks, values)}
            earliest = int(weeks.min())

            x = np.empty(self.lookback, dtype=float)
            ok = True
            for i, ww in enumerate(win_weeks):
                if ww in mp:
                    x[i] = mp[ww]
                else:
                    if self.fill_missing == 'zero':
                        x[i] = 0.0
                    elif self.fill_missing == 'nan':
                        x[i] = np.nan
                    else:  # ffill: 직전 관측으로 채움(없으면 0)
                        prev = ww
                        found = False
                        for _ in range(self.target_horizon):
                            prev = DateUtil.add_weeks_yyyyww(prev, -1)  # 1주 뒤로
                            if prev < earliest:
                                break
                            if prev in mp:
                                x[i] = mp[prev]
                                found = True
                                break
                        if not found:
                            x[i] = 0.0

            if self.fill_missing == 'nan' and not np.any(np.isfinite(x)):
                ok = False
            if not ok:
                continue

            self.inputs.append(x)
            self.part_ids.append(part)
            self.hist_yyyyww.append(list(win_weeks))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # (L,1) 텐서와 part_id 반환
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(-1),
            self.part_ids[idx]
        )


class MultiPartAnchoredInferenceByYYYYMM(Dataset):
    """
        df: pl.DataFrame({'oper_part_no': str, 'demand_dt': int(YYYYMM), 'demand_qty': float})
        plan_yyyymm: int (예: 202210). '그 이후부터' 예측하고자 한다면,
          - 입력은 plan_yyyymm '직전' L개월
          - 첫 예측 달력은 보통 plan_yyyymm (포함) 또는 plan+1 (이후) → 플롯/축에서 결정
        fill_missing: 'ffill' | 'zero' | 'nan'
          - 윈도우 내 존재하지 않는 월을 어떻게 채울지
        parts_filter: 특정 part만 추론하고 싶을 때 리스트/셋 전달
    """
    def __init__(self,
                 df: pl.DataFrame,
                 lookback: int,
                 plan_yyyymm: int,
                 part_col: str = 'oper_part_no',
                 date_col: str = 'demand_dt',
                 qty_col: str = 'demand_qty',
                 parts_filter = None,
                 fill_missing: str = 'ffill',
                 target_horizon: int = 120,
                 ):
        self.lookback = int(lookback)
        self.plan_yyyymm = int(plan_yyyymm)
        self.part_col = part_col
        self.date_col = date_col
        self.qty_col = qty_col
        assert fill_missing in ('ffill', 'zero', 'nan')
        self.fill_missing = fill_missing

        self.inputs: list[np.ndarray] = [] # [L]
        self.part_ids: list[str] = [] # part_no
        self.hist_yyyymm: list[list[int]] = []
        self.target_horizon = target_horizon

        parts_set = set(parts_filter) if parts_filter is not None else None
        grouped = df.partition_by(part_col)

        for g in grouped:
            part = g[part_col][0]
            if parts_set is not None and part not in parts_set:
                continue

            gd = g.select([date_col, qty_col]).sort(date_col)
            months = gd[date_col].to_numpy().astype(np.int64)
            values = gd[qty_col].to_numpy().astype(float)

            if len(months) == 0:
                continue

            # 입력 윈도우 달력 (앵커 '직전' L개월)
            win_months = DateUtil.month_seq_ending_before(self.plan_yyyymm, self.lookback) # [L]
            #  월 value dictionary
            mp = {int(m): float(v) for m, v in zip(months, values)}
            earliest = int(months.min())

            # fill null
            x = np.empty(self.lookback, dtype = float)
            ok = True
            for i, mm in enumerate(win_months):
                if mm in mp:
                    x[i] = mp[mm]

                else:
                    if self.fill_missing == 'zero':
                        x[i] = 0.0
                    elif self.fill_missing == 'nan':
                        x[i] = np.nan
                    else: # ffill: 직전 관측으로 채움 (없으면 0)
                        prev = mm
                        found = False
                        # max 120 month
                        for _ in range(self.target_horizon):
                            prev = DateUtil.add_months_yyyymm(prev, -1)
                            if prev < earliest:
                                break
                            if prev in mp:
                                x[i] = mp[prev]
                                found = True
                                break
                        if not found:
                            x[i] = 0.0 # 완전 무관측이면 0으로..
            if self.fill_missing == 'nan' and not np.any(np.isfinite(x)):
                ok = False

            if not ok:
                continue

            self.inputs.append(x)
            self.part_ids.append(part)
            self.hist_yyyymm.append(list(win_months))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype = torch.float32).unsqueeze(-1),  # [L, 1]
            self.part_ids[idx]
        )

class MultiPartDataModule:
    """
        모델 학습용 멀티파트 시계열 데이터 모듈
        - 학습/검증용 DataLoader 생성
        - Inference용 DataLoader 생성
        - 내부적으로 config 객체를 활용 (lookback, horizon 등)
    """
    def __init__(self,
                 df: pl.DataFrame,
                 config,
                 is_running,
                 batch_size = 64,
                 val_ratio = 0.2,
                 shuffle = True,
                 seed = 42):
        self.df = df
        self.config = config
        self.is_running = is_running
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.inference_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.inference_loader = None

    def setup(self):
        '''
            학습/검증 Dataset 분할 및 생성
        '''

        full_dataset = MultiPartTrainingDataset(self.df, self.config)
        total_len = len(full_dataset)
        val_len = int(total_len * self.val_ratio)
        train_len = total_len - val_len

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_len, val_len], generator = generator)

    def get_train_loader(self):
        if self.train_dataset is None:
            self.setup()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            drop_last = True
        )
        return self.train_loader

    def get_val_loader(self):
        if self.val_dataset is None:
            self.setup()

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            drop_last = False
        )
        return self.val_loader

    def get_inference_loader(self):
        '''
            inference용 전체 파트별 입력 시퀀스 DataLoader 생성
        '''
        self.inference_dataset = MultiPartInferenceDataset(self.df, self.config)
        self.inference_loader = DataLoader(
            self.inference_dataset,
            batch_size = self.batch_size,
            shuffle = False
        )
        return self.inference_loader

    def get_inference_loader_at_plan(self, plan_dt: int, parts_filter = None, fill_missing: str = 'ffill'):
        if self.is_running:
            ds = MultiPartAnchoredInferenceByYYYYWW(
                df = self.df,
                lookback = self.config.lookback,
                plan_yyyyww = plan_dt,
                part_col = 'oper_part_no',
                date_col = 'demand_dt',
                qty_col = 'demand_qty',
                parts_filter = parts_filter,
                fill_missing = fill_missing
            )

        else:
            ds = MultiPartAnchoredInferenceByYYYYMM(
                df = self.df,
                lookback = self.config.lookback,
                plan_yyyymm = plan_dt,
                part_col = 'oper_part_no',
                date_col = 'demand_dt',
                qty_col = 'demand_qty',
                parts_filter = parts_filter,
                fill_missing = fill_missing
            )

        return DataLoader(ds, batch_size = self.batch_size, shuffle = False)


