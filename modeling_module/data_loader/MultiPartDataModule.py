import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

from modeling_module.utils.date_util import DateUtil


class MultiPartInferenceDataset(Dataset):
    """
        Multi-part 시계열 데이터를 추론(inference)용으로 구성하기 위한 PyTorch Dataset 클래스.

        각 부품(part)별로 시계열 데이터를 분리한 뒤,
        마지막 lookback 구간만 잘라서 입력 시퀀스로 구성한다.
        주로 모델 추론 시 과거 데이터를 기반으로 예측할 때 사용된다.
    """
    def __init__(self,
                 df: pl.DataFrame,
                 lookback: int = None,
                 part_col: str = 'oper_part_no',
                 date_col: str = 'demand_dt',  # 주차 컬럼명(정수 YYYYWW)
                 qty_col: str = 'demand_qty',
                 ):
        """
            Dataset 초기화 및 부품별 시계열 입력 구성

            Parameters:
            - df (pl.DataFrame): 'oper_part_no'와 'demand_dt', 'demand_qty' 컬럼을 포함하는 전체 시계열 데이터.
            - config: lookback 설정 값을 포함하는 설정 객체.
        """
        assert lookback is not None
        self.lookback = lookback # 입력 시퀀스 길이
        self.inputs = []                # 최종 모델 입력 시퀀스 리스트 (numpy array)
        self.part_ids = []              # 각 시퀀스에 대응하는 부품 식별자 리스트 (string)

        # 부품별로 데이터 분할
        grouped = df.partition_by(part_col)
        for g in grouped:
            series = g.sort(date_col)[qty_col].to_numpy()
            part_no = g[part_col][0]

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
    """
        다중 부품 시계열 데이터를 학습용으로 구성하는 PyTorch Dataset 클래스.

        - 각 부품별 시계열 데이터를 기반으로 슬라이딩 윈도우 방식으로
          (lookback, horizon) 형태의 훈련 샘플을 생성한다.
        - 모델은 lookback만큼의 과거 데이터를 입력으로 받아 horizon 만큼의 미래를 예측한다.
    """
    def __init__(self,
                 df: pl.DataFrame,
                 lookback: int = None, horizon: int = None,
                 part_col: str = 'oper_part_no',
                 date_col: str = 'demand_dt',  # 주차 컬럼명(정수 YYYYWW)
                 qty_col: str = 'demand_qty',
                 ):

        assert lookback, horizon is not None
        self.lookback = lookback # 입력 구간 길이
        self.horizon = horizon   # 예측 구간 길이

        self.samples = []               # 최종 (x_seq, y_seq) 튜플 목록
        self.part_ids = []              # 각 시퀀스에 대응하는 부품 ID

        # 부품별로 데이터 분할
        grouped = df.partition_by(part_col)

        for g in grouped:
            series = g.sort(date_col)[qty_col].to_numpy()
            part_no = g[part_col][0]

            # 최소 샘플 조건 확인: lookback + horizon 이상 길이만 처리
            if len(series) < self.lookback + self.horizon:
                continue
            for i in range(len(series) - self.lookback - self.horizon + 1):
                x_seq = series[i: i+self.lookback] # 과거 입력 시퀀스
                y_seq = series[i+self.lookback: i+self.lookback+self.horizon] # 미래 예측 대상
                self.samples.append((x_seq, y_seq))
                self.part_ids.append(part_no)

    def __len__(self):
        """
        전체 훈련 샘플 개수 반환
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
                인덱스에 해당하는 샘플을 PyTorch 텐서 형태로 반환

                Returns:
                - x_seq (torch.Tensor): [lookback, 1] 형태의 입력 시퀀스
                - y_seq (torch.Tensor): [horizon] 형태의 타깃 시퀀스
                - part_id (str): 해당 샘플의 부품 ID
        """
        x_seq, y_seq = self.samples[idx]
        return (
            torch.tensor(x_seq, dtype = torch.float32).unsqueeze(-1),
            torch.tensor(y_seq, dtype = torch.float32),
            self.part_ids[idx]
        )

class MultiPartAnchoredInferenceByYYYYWW(Dataset):
    """
    주차(YYYYWW) 기반 앵커드(anchor-based) 추론을 위한 PyTorch Dataset 클래스.

    각 부품별로 특정 주차(plan_yyyyww)를 기준(anchor point)으로 하여,
    그 이전 lookback 주의 데이터를 구성하여 추론 입력으로 사용한다.

    주요 기능:
    - 주차 누락 시 결측값 처리 방법(filled with ffill / zero / nan) 선택 가능
    - 특정 부품(part)만 대상으로 필터링 가능
    - lookback 주차 내 일부 또는 전체 결측 시 fallback 로직 포함
    """

    def __init__(self,
                 df: pl.DataFrame,
                 lookback: int,
                 plan_yyyyww: int,
                 part_col: str = 'oper_part_no',
                 date_col: str = 'demand_dt',     # 주차 컬럼명
                 qty_col: str = 'demand_qty',
                 parts_filter=None,
                 fill_missing: str = 'ffill',
                 target_horizon: int = 104):       # 2년치 백업 탐색 기본
        """
                Dataset 초기화: 부품별로 anchor 이전 lookback 주차의 수요 시퀀스를 구성

                Parameters:
                - df: 입력 시계열 데이터 (oper_part_no, demand_dt, demand_qty 포함)
                - lookback: 과거 입력 길이 (주 단위)
                - plan_yyyyww: 기준 주차 (예측 시작 기준)
                - part_col: 부품 컬럼명
                - date_col: 주차 컬럼명 (예: demand_dt → 202252)
                - qty_col: 수요량 컬럼명
                - parts_filter: 추론 대상 부품 목록 (리스트 또는 셋)
                - fill_missing: 결측 주차 처리 방법 ('ffill' | 'zero' | 'nan')
                - target_horizon: ffill 시 백트래킹 허용 최대 주 수 (예: 104주 = 2년)
        """
        self.lookback = int(lookback)
        self.plan_yyyyww = int(plan_yyyyww)
        self.part_col = part_col
        self.date_col = date_col
        self.qty_col = qty_col
        assert fill_missing in ('ffill', 'zero', 'nan')
        self.fill_missing = fill_missing
        self.target_horizon = int(target_horizon)

        self.inputs: list[np.ndarray] = []     # (lookback, ) 시계열 입력
        self.part_ids: list[str] = []          # 각 부품의 id
        self.hist_yyyyww: list[list[int]] = [] # 입력 주차 리스트 (ex: [202251, 202252, ...])

        parts_set = set(parts_filter) if parts_filter is not None else None

        # 파트별 그룹
        grouped = df.partition_by(part_col)

        for g in grouped:
            part = g[part_col][0]
            if parts_set is not None and part not in parts_set:
                continue

            # 시계열 정렬 및 주차, 수요 추출
            gd = g.select([date_col, qty_col]).sort(date_col)
            weeks = gd[date_col].to_numpy().astype(np.int64)   # YYYYWW (ISO week)
            values = gd[qty_col].to_numpy().astype(float)

            if len(weeks) == 0:
                continue

            # 기준 주차 이전 L주 리스트 생성
            win_weeks = DateUtil.week_seq_ending_before(self.plan_yyyyww, self.lookback)  # 길이 L
            mp = {int(w): float(v) for w, v in zip(weeks, values)}
            earliest = int(weeks.min())

            # 입력 시퀀스 채우기 (결측 주차 포함)
            x = np.empty(self.lookback, dtype=float)
            ok = True
            for i, ww in enumerate(win_weeks):
                if ww in mp:
                    x[i] = mp[ww]
                else:
                    # 결측 주차 처리
                    if self.fill_missing == 'zero':
                        x[i] = 0.0
                    elif self.fill_missing == 'nan':
                        x[i] = np.nan
                    else:  # ffill: 직전 관측으로 채움(없으면 0)
                        prev = ww
                        found = False
                        # ⏮과거로 최대 target_horizon 주차까지 백트래킹
                        for _ in range(self.target_horizon):
                            prev = DateUtil.add_weeks_yyyyww(prev, -1)  # 1주 뒤로
                            if prev < earliest:
                                break
                            if prev in mp:
                                x[i] = mp[prev]
                                found = True
                                break
                        if not found:
                            x[i] = 0.0 # 백트래킹 실패 시 0으로 대체

            # 'nan' 처리 시 유효한 수치가 하나도 없으면 skip
            if self.fill_missing == 'nan' and not np.any(np.isfinite(x)):
                ok = False
            if not ok:
                continue

            # 최종 시퀀스 저장
            self.inputs.append(x)
            self.part_ids.append(part)
            self.hist_yyyyww.append(list(win_weeks))

    def __len__(self):
        """
            전체 부품 수 (입력 시퀀스 수) 반환
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
            idx번째 부품의 입력 시퀀스 및 부품 ID 반환

            Returns:
            - x: [lookback, 1] 형태의 텐서 (PyTorch 모델 입력용)
            - part_id: 부품 식별자
        """
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(-1),
            self.part_ids[idx]
        )


class MultiPartAnchoredInferenceByYYYYMM(Dataset):
    """
        월 단위(YYYYMM) 시계열 데이터를 기반으로 한 앵커드(anchor-based) 추론용 Dataset 클래스.

        - 기준 달(plan_yyyymm)을 앵커로 하여, 그 이전 L개월을 입력으로 사용
        - 각 부품(part)별로 독립적인 시계열을 구성
        - 결측 월 처리 방식(filled with ffill / zero / nan) 선택 가능
        - 특정 부품 필터링, 백트래킹 거리 설정 등 고급 설정 가능

        사용 시점:
        - 예측 기준 월 이전의 데이터만 사용해야 할 경우
        - 입력 시계열이 월별 단위로 구성된 모델에 추론을 수행할 경우
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
        """
            Dataset 초기화: 부품별로 anchor 이전 lookback 주차의 수요 시퀀스를 구성

            Parameters:
            - df: Polars DataFrame. 필수 컬럼: ['oper_part_no', 'demand_dt', 'demand_qty']
            - lookback: 입력 시퀀스 길이 (월 수)
            - plan_yyyymm: 앵커 기준 월 (예: 202210)
            - part_col: 부품 식별 컬럼명
            - date_col: 월 컬럼명 (정수형 YYYYMM)
            - qty_col: 수요량 컬럼명
            - parts_filter: 특정 부품만 대상으로 추론하고자 할 경우 리스트/셋 지정
            - fill_missing: 결측 월 처리 방법 ('ffill' | 'zero' | 'nan')
            - target_horizon: ffill 적용 시 백트래킹 허용 최대 개월 수 (기본: 120개월)
        """

        self.lookback = int(lookback)
        self.plan_yyyymm = int(plan_yyyymm)
        self.part_col = part_col
        self.date_col = date_col
        self.qty_col = qty_col
        assert fill_missing in ('ffill', 'zero', 'nan')
        self.fill_missing = fill_missing
        self.target_horizon = target_horizon

        self.inputs: list[np.ndarray] = []      # 각 부품의 입력 시계열 [lookback]
        self.part_ids: list[str] = []           # 각 시퀀스의 부품 ID
        self.hist_yyyymm: list[list[int]] = []  # 각 부품의 입력 시퀀스 월 리스트

        parts_set = set(parts_filter) if parts_filter is not None else None

        # 부품별로 데이터를 그룹화
        grouped = df.partition_by(part_col)

        for g in grouped:
            part = g[part_col][0]
            if parts_set is not None and part not in parts_set:
                continue

            # 시계열 정렬 및 수요량 추출
            gd = g.select([date_col, qty_col]).sort(date_col)
            months = gd[date_col].to_numpy().astype(np.int64)
            values = gd[qty_col].to_numpy().astype(float)

            if len(months) == 0:
                continue

            # 입력 윈도우 달력 (앵커 '직전' L개월)
            win_months = DateUtil.month_seq_ending_before(self.plan_yyyymm, self.lookback) # [L]

            # {yyyymm: demand_qty} 딕셔너리
            mp = {int(m): float(v) for m, v in zip(months, values)}
            earliest = int(months.min())

            # 입력 시퀀스 초기화 및 결측 처리
            x = np.empty(self.lookback, dtype = float)
            ok = True
            for i, mm in enumerate(win_months):
                if mm in mp:
                    x[i] = mp[mm]

                else:
                    # 결측 월 처리 방식
                    if self.fill_missing == 'zero':
                        x[i] = 0.0
                    elif self.fill_missing == 'nan':
                        x[i] = np.nan
                    else: # ffill: 직전 관측으로 채움 (없으면 0)
                        prev = mm
                        found = False
                        # max 120 month
                        for _ in range(self.target_horizon): # 최대 N개월 백트래킹
                            prev = DateUtil.add_months_yyyymm(prev, -1)
                            if prev < earliest:
                                break
                            if prev in mp:
                                x[i] = mp[prev]
                                found = True
                                break
                        if not found:
                            x[i] = 0.0 # 완전 무관측이면 0으로..

            # 'nan' 처리 시 유효한 수치가 하나도 없으면 skip
            if self.fill_missing == 'nan' and not np.any(np.isfinite(x)):
                ok = False
            if not ok:
                continue

            # 최종 시퀀스 저장
            self.inputs.append(x)
            self.part_ids.append(part)
            self.hist_yyyymm.append(list(win_months))

    def __len__(self):
        """
            전체 부품 수 (입력 시퀀스 수) 반환
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
            idx번째 부품의 입력 시퀀스 및 부품 ID 반환

            Returns:
            - x: [lookback, 1] 형태의 텐서 (PyTorch 모델 입력용)
            - part_id: 부품 식별자
        """
        return (
            torch.tensor(self.inputs[idx], dtype = torch.float32).unsqueeze(-1),  # [L, 1]
            self.part_ids[idx]
        )

class MultiPartDataModule:
    """
    멀티파트 시계열 학습/추론을 위한 데이터 모듈 클래스

    - 학습/검증용 Dataset 분할 및 DataLoader 생성
    - 추론용 DataLoader (일반 + 앵커기준) 생성
    - 내부적으로 config 객체를 참조하여 lookback/horizon 등 사용

    주요 사용 시나리오:
    - 모델 학습 시 train/val 로더 자동 세팅
    - 특정 시점(plan_dt)에 대한 시계열 예측용 입력 자동 구성
    """
    def __init__(self,
                 df: pl.DataFrame,
                 lookback: int,
                 horizon: int,
                 is_running,
                 batch_size = 64,
                 val_ratio = 0.2,
                 shuffle = True,
                 seed = 42):
        """
        생성자: 주요 설정 값 초기화

        Parameters:
        - df: 전체 시계열 데이터 (polars DataFrame)
        - config: 학습 및 추론에 필요한 하이퍼파라미터 객체 (lookback, horizon 등 포함)
        - is_running: True → 주차 기준(YYYYWW), False → 월 기준(YYYYMM)
        - batch_size: DataLoader 배치 크기
        - val_ratio: 검증 데이터 비율
        - shuffle: 학습 데이터 셔플 여부
        - seed: 고정 시드를 통한 재현성 보장
        """
        self.df = df
        self.lookback = lookback
        self.horizon = horizon
        self.is_running = is_running
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.seed = seed

        # 내부 변수 초기화
        self.train_dataset = None
        self.val_dataset = None
        self.inference_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.inference_loader = None

    def setup(self):
        """
        학습 및 검증용 Dataset 생성 및 분할

        - 전체 dataset을 생성한 후, train/val로 비율에 따라 분할
        - 동일 시드로 항상 동일하게 분할됨
        """
        full_dataset = MultiPartTrainingDataset(self.df, self.lookback, self.horizon)
        total_len = len(full_dataset)
        val_len = int(total_len * self.val_ratio)
        train_len = total_len - val_len

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_len, val_len], generator = generator)

    def get_train_loader(self):
        """
        학습용 DataLoader 반환
        """
        if self.train_dataset is None:
            self.setup()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            drop_last = True    # 마지막 미니배치 제외 (정규 학습 안정성 확보 목적)
        )
        return self.train_loader

    def get_val_loader(self):
        """
        검증용 DataLoader 반환
        """
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
        """
        단순 추론용 DataLoader 반환
        - 가장 최근 lookback 데이터를 사용하여 추론 입력 구성
        - 학습이 아닌 전용 추론에 사용
        """
        self.inference_dataset = MultiPartInferenceDataset(self.df, self.lookback)
        self.inference_loader = DataLoader(
            self.inference_dataset,
            batch_size = self.batch_size,
            shuffle = False
        )
        return self.inference_loader

    def get_inference_loader_at_plan(self, plan_dt: int, parts_filter = None, fill_missing: str = 'ffill'):
        """
        특정 시점(plan_dt)에 대한 앵커 기반 추론 입력 DataLoader 생성

        Parameters:
        - plan_dt: 기준 시점 (YYYYWW or YYYYMM)
        - parts_filter: 부품 필터링 리스트 또는 셋
        - fill_missing: 결측 데이터 처리 방식 ('ffill' | 'zero' | 'nan')

        Returns:
        - DataLoader: 앵커 기반 추론 입력용 DataLoader
        """

        if self.is_running:
            ds = MultiPartAnchoredInferenceByYYYYWW(
                df = self.df,
                lookback = self.lookback,
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
                lookback = self.lookback,
                plan_yyyymm = plan_dt,
                part_col = 'oper_part_no',
                date_col = 'demand_dt',
                qty_col = 'demand_qty',
                parts_filter = parts_filter,
                fill_missing = fill_missing
            )

        return DataLoader(ds, batch_size = self.batch_size, shuffle = False)
