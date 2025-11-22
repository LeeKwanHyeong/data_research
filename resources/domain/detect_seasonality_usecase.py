
from __future__ import annotations

from typing import Optional, List, Dict, Any

import polars as pl



# ==========================================
# 1. 유틸: phase / center / 기본 period 생성
# ==========================================

def _center_of_range(r: range) -> int:
    """
    range(start, stop) 에 대해, 포함 구간 [start, stop-1] 의 중앙값을 반환.
    예: range(9,22) -> 9~21 → center ≒ 15
    """
    return (r.start + (r.stop - 1)) // 2


def build_default_periods_from_season_range(
    season_range: Dict[str, Any],
    *,
    max_years: int = 2,
) -> List[int]:
    """
    season_range (예: season_V101_range) 를 받아
      - 각 시즌 label(S_SP, S_SM, S_AU, S_WI, 또는 S_M, S_SM_D 등)에 대한
        대표 phase(1~52)를 구하고,
      - 1년~max_years년까지의 lag 후보들을 DEFAULT_PERIODS로 생성한다.

    ※ 대표 phase:
       - range인 경우: 중앙값
       - [range, range, ...] 리스트인 경우: 각 range의 중앙값 모두 사용
         (원하면 하나만 쓰도록 바꿀 수도 있음)
    """
    phases: List[int] = []

    for _label, ranges in season_range.items():
        if isinstance(ranges, range):
            centers = [_center_of_range(ranges)]
        else:
            # S_WI 처럼 [range, range, ...] 구조
            centers = [_center_of_range(r) for r in ranges]

        phases.extend(centers)

    # 중복 제거
    phases = sorted(set(phases))

    periods: List[int] = []
    for year in range(1, max_years + 1):
        for ph in phases:
            periods.append(52 * year + ph)

    return sorted(set(periods))


def _week_to_phase(yyyyww: int) -> int:
    """
    yyyyww → 연내 주차 phase(1~52)로 변환.
    52주 기준, % 52 해서 0이면 52로 본다.
    """
    phase = yyyyww % 52
    return 52 if phase == 0 else phase


# ==========================================
# 2. UseCase 본체
# ==========================================
season_V101_range = { 'S_SP': range(9, 22), 'S_SM': range(22, 36), 'S_AU': range(36, 49), 'S_WI': [range(49, 53), range(1, 9)]}
season_V403_range = { 'S_SP': range(9, 22), 'S_SM': range(22, 36), 'S_AU': range(36, 49), 'S_WI': [range(49, 53), range(1, 9)]}
season_V401_range = { 'S_SP': range(9, 22), 'S_SM': range(22, 36), 'S_AU': range(36, 49), 'S_WI': [range(49, 53), range(1, 9)]}
season_V506_range = { 'S_M': range(4, 8), 'S_SM_D': range(9, 22), 'S_SM_W': range(22, 40), 'S_AU': range(40, 49), 'S_WI': [range(49, 52), range(1, 9)]}

def get_seasonality_type(plant_cd: str):
    if plant_cd == 'V101': return season_V101_range
    elif plant_cd == 'V403': return season_V403_range
    elif plant_cd == 'V401': return season_V401_range
    elif plant_cd == 'V506': return season_V506_range
    else: season_V101_range


class DetectSeasonalityUseCase:
    """
    주문 주차(order_yyyyww) 기반으로 부품별 Seasonality를 탐지하는 유즈케이스.

    - 부품별 시작 주차 상이 → 전체 공통 주차 그리드 위에서 0-fill
    - period별 lag-corr 계산
    - plant_cd별 season_range 에 따라 시즌별 score 집계
    - 시즌 label은 season_config에 정의된 대로 사용 (S_SP, S_SM, S_AU, S_WI,
      또는 S_M, S_SM_D, S_SM_W 등 어떤 label이든 처리 가능)
    """

    def __init__(
        self,
        target_df: pl.DataFrame,
        time_col: str = "order_yyyyww",
        target_col: str = "demand_qty",
        part_col: str = "oper_part_no",
        periods: Optional[List[int]] = None,
        threshold: float = 0.5,
        min_cycles: int = 2,
        plant_cd: Optional[str] = None,
        max_years_for_default_periods: int = 2,
    ):
        """
        Parameters
        ----------
        target_df : pl.DataFrame
            최소 [part_col, time_col, target_col]을 포함하는 데이터프레임.
            예: ['oper_part_no', 'order_yyyyww', 'demand_qty']

        time_col : str
            주차/월 컬럼명 (yyyyww 또는 yyyymm 등). 여기선 주로 yyyyww 가정.

        target_col : str
            수요량 컬럼명.

        part_col : str
            부품 ID 컬럼명.

        periods : Optional[List[int]]
            corr(y_t, y_{t-p}) 를 계산할 lag 후보 리스트.
            None이면 plant_cd에 해당하는 season_range에서 자동으로 생성.

        threshold : float
            시즌 score (시즌별 max abs corr)가 이 값 이상일 때만 Seasonality 인정.

        min_cycles : int
            특정 period p에 대해 n_obs >= p * min_cycles 조건을 만족하는 경우만
            corr를 유효로 봄. 그렇지 않으면 그 period의 corr는 0으로 마스킹.

        plant_cd : Optional[str]
            V101 / V403 / V401 / V506 등.
            - season_config.get_seasonality_type(plant_cd) 로 season_range를 얻음.
            - periods가 None이면, 해당 season_range로부터 기본 periods를 생성.

        max_years_for_default_periods : int
            periods를 자동 생성할 때 사용할 최대 연수 (기본 2년 주기까지).
        """
        self.target_df = target_df
        self.part_col = part_col
        self.time_col = time_col
        self.target_col = target_col
        self.threshold = threshold
        self.min_cycles = min_cycles
        self.plant_cd = plant_cd

        # 1) plant_cd별 시즌 구간 (phase 기반)
        self.season_range: Dict[str, Any] = get_seasonality_type(plant_cd or "V101")

        # 2) periods 설정
        if periods is not None:
            self.periods = periods
        else:
            # season_range에서 자동으로 lag 후보 생성
            self.periods = build_default_periods_from_season_range(
                self.season_range,
                max_years=max_years_for_default_periods,
            )

    # --------------------------------------
    # (1) 부품별/전체 공통 시점 그리드 생성 + 0 fill
    # --------------------------------------
    def _fill_missing_periods(self) -> pl.DataFrame:
        """
        target_df에 존재하는 모든 부품 × 모든 time_col 의 그리드를 만든 뒤,
        존재하지 않는 시점은 target_col = 0.0 으로 채운 DataFrame을 반환.
        """
        df = self.target_df

        parts_df = df.select(self.part_col).unique()
        periods_df = df.select(self.time_col).unique().sort(self.time_col)

        full_grid = parts_df.join(periods_df, how="cross")

        df_filled = (
            full_grid
            .join(df, on=[self.part_col, self.time_col], how="left")
            .with_columns(
                pl.col(self.target_col)
                  .fill_null(0.0)
                  .cast(pl.Float64)
                  .alias(self.target_col)
            )
            .sort([self.part_col, self.time_col])
        )
        return df_filled

    # --------------------------------------
    # (2) periods → 시즌별 period 리스트 매핑
    #     (라벨이 무엇이든 상관없이 처리)
    # --------------------------------------
    def _build_season_to_periods(self) -> Dict[str, List[int]]:
        """
        self.periods (lag, 예: 67, 80, 94, 104...) 를
        self.season_range (phase 기반, 예: S_SP: 9~21, ...) 에 따라
        시즌별로 모아주는 역할.

        - phase = period % 52 (0이면 52)
        - 그 phase가 어느 시즌 range에 속하는지 검사
        """
        season_to_periods: Dict[str, List[int]] = {
            label: [] for label in self.season_range.keys()
        }

        for p in self.periods:
            phase = p % 52
            if phase == 0:
                phase = 52

            for label, ranges in self.season_range.items():
                if isinstance(ranges, range):
                    if phase in ranges:
                        season_to_periods[label].append(p)
                else:
                    # 리스트(복수 range)인 경우
                    for r in ranges:
                        if phase in r:
                            season_to_periods[label].append(p)

        return season_to_periods

    # --------------------------------------
    # (3) 멀티 period 기반 Seasonality 탐지 (구간-based)
    # --------------------------------------
    def detect_multi_period_seasonality(self) -> pl.DataFrame:
        """
        최종 출력:
            [part_col, n_obs, max_season_score, season_flag, season_type,
             actual_order_count, <각 시즌별 score 컬럼 ...>]
        """
        # 1) 결측 기간 0-fill 후 정렬
        df = self._fill_missing_periods()
        df_sorted = df.sort([self.part_col, self.time_col])

        # 2) period별 lag-corr 집계 준비
        aggs = [pl.len().alias("n_obs")]
        for p in self.periods:
            aggs.append(
                pl.corr(
                    pl.col(self.target_col),
                    pl.col(self.target_col).shift(p),
                ).alias(f"corr_lag_{p}")
            )

        # 3) 부품별 corr 및 관측 개수 집계
        stats = df_sorted.group_by(self.part_col, maintain_order=True).agg(aggs)

        # 4) abs(corr) 생성
        for p in self.periods:
            stats = stats.with_columns(
                pl.col(f"corr_lag_{p}").abs().alias(f"abs_corr_lag_{p}")
            )

        # 5) NaN/Null 방어
        stats = stats.fill_nan(0.0).fill_null(0.0)

        # 6) period별 데이터 길이 조건 반영
        #    n_obs < p * min_cycles 인 period의 corr는 0으로 마스킹
        for p in self.periods:
            stats = stats.with_columns(
                pl.when(pl.col("n_obs") >= p * self.min_cycles)
                  .then(pl.col(f"abs_corr_lag_{p}"))
                  .otherwise(0.0)
                  .alias(f"abs_corr_lag_{p}")
            )

        # 7) 시즌별 포함 period 매핑
        #    (label → [period...])
        season_to_periods = self._build_season_to_periods()

        # 8) 시즌별 score 계산: 각 시즌에 속하는 period들의 abs_corr 중 최대값
        season_score_cols: List[str] = []
        for label, period_list in season_to_periods.items():
            col_name = f"{label}_score"
            if period_list:
                cols = [pl.col(f"abs_corr_lag_{p}") for p in period_list]
                stats = stats.with_columns(
                    pl.max_horizontal(cols).alias(col_name)
                )
            else:
                stats = stats.with_columns(
                    pl.lit(0.0).alias(col_name)
                )
            season_score_cols.append(col_name)

        # 9) 시즌 score 중 최대값 및 그 시즌 레이블 찾기
        if season_score_cols:
            stats = stats.with_columns(
                pl.max_horizontal([pl.col(c) for c in season_score_cols]).alias(
                    "max_season_score"
                )
            )
        else:
            stats = stats.with_columns(
                pl.lit(0.0).alias("max_season_score")
            )

        def _pick_best_season(row: Dict[str, Any]) -> str:
            """
            각 row의 시즌 score들 중 max_season_score와 같은 첫 시즌을 선택.
            max_season_score <= threshold 이면 Non-Seasonal.
            """
            max_score = row.get("max_season_score", 0.0)
            if max_score <= 0:
                return "Non-Seasonal"

            if max_score <= self.threshold:
                return "Non-Seasonal"

            for col_name in season_score_cols:
                if abs(row.get(col_name, 0.0) - max_score) < 1e-12:
                    # 'S_SP_score' -> 'S_SP'
                    return col_name.replace("_score", "")
            return "Non-Seasonal"

        stats = stats.with_columns(
            pl.struct(season_score_cols + ["max_season_score"])
              .map_elements(_pick_best_season)
              .alias("season_type")
        )

        # 10) season_flag: 시즌성이 있다고 볼지 여부
        stats = stats.with_columns(
            pl.when(
                (pl.col("max_season_score") > self.threshold)
                & (pl.col("season_type") != "Non-Seasonal")
            )
            .then(pl.lit("Y"))
            .otherwise(pl.lit("N"))
            .alias("season_flag")
        )

        # 11) 실제 주문 주차 개수 (원본 기준, 0-fill 전)
        actual_order_count = (
            self.target_df
            .group_by(self.part_col)
            .agg(
                pl.col(self.time_col)
                  .n_unique()
                  .alias("actual_order_count")
            )
        )
        stats = stats.join(actual_order_count, on=self.part_col, how="inner")

        # 12) 결과 선택
        select_cols = [
            self.part_col,
            "n_obs",
            "max_season_score",
            "season_flag",
            "season_type",
            "actual_order_count",
        ] + season_score_cols

        return stats.select(select_cols)