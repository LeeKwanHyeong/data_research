from __future__ import annotations
from dataclasses import dataclass
import polars as pl


@dataclass
class IntermittentConfig:
    name_col: str = "oper_part_no"
    target_col: str = "demand_qty"

    # ADI / CV² 기준값 (문헌에서 자주 쓰는 값)
    adi_threshold: float = 1.32       # ADI 기준: 크면 간헐(Intermittent/Lumpy) 경향
    cv2_threshold: float = 0.49       # CV^2 기준: 크면 변동성이 큰 수요

    # "거의 0"으로 볼 수요 기준 (예: 20 → 20 미만은 사실상 0으로 취급)
    count_threshold: float = 0.0

    # 간헐 여부(is_sparsity) 판정 시 CV²도 같이 쓸지 여부
    use_cv2: bool = False

    # 최소 히스토리 길이 (이보다 짧으면 type을 'insufficient'으로 둠)
    min_periods: int = 10


class IntermittentDemandDetector:
    """
    주간 수요 테이블에서 각 부품의
    - 간헐 수요 여부(is_sparsity)
    - 수요 유형(smooth / erratic / intermittent / lumpy / insufficient)
    를 판별하는 클래스.
    """

    def __init__(self, weekly_df: pl.DataFrame, config: IntermittentConfig | None = None):
        self.weekly_df = weekly_df
        self.config = config or IntermittentConfig()

        missing = [
            c for c in (self.config.name_col, self.config.target_col)
            if c not in self.weekly_df.columns
        ]
        if missing:
            raise ValueError(f"DataFrame에 {missing} 컬럼이 없습니다.")

    # --------------------------------
    # 내부: 통계량 계산 (ADI / CV² 등)
    # --------------------------------
    def _compute_stats(self) -> pl.DataFrame:
        cfg = self.config
        name_col = cfg.name_col
        target_col = cfg.target_col

        thr = cfg.count_threshold  # "거의 0"으로 볼 기준

        stats = (
            self.weekly_df
            .group_by(name_col)
            .agg(
                # 전체 주차 수
                pl.len().alias("n_periods"),

                # "거의 없음"으로 보는 주차 수 (target < thr)
                pl.col(target_col)
                  .filter(pl.col(target_col) < thr)
                  .count()
                  .alias("n_zero"),

                # "수요 발생"으로 보는 주차 수 (target >= thr)
                pl.col(target_col)
                  .filter(pl.col(target_col) >= thr)
                  .count()
                  .alias("n_nz"),

                # 수요 발생 주의 평균/표준편차
                pl.col(target_col)
                  .filter(pl.col(target_col) >= thr)
                  .mean()
                  .alias("nz_mean"),

                pl.col(target_col)
                  .filter(pl.col(target_col) >= thr)
                  .std()
                  .alias("nz_std"),
            )
            .with_columns(
                # 0(또는 거의 0) 비율
                (pl.col("n_zero") / pl.col("n_periods")).alias("zero_ratio"),

                # ADI = 전체 주차 수 / (수요 발생 주차 수)
                pl.when(pl.col("n_nz") > 0)
                  .then(pl.col("n_periods") / pl.col("n_nz"))
                  .otherwise(None)
                  .alias("ADI"),

                # CV^2 = (sigma / mu)^2
                pl.when((pl.col("nz_mean") > 0) & pl.col("nz_std").is_not_null())
                  .then((pl.col("nz_std") / pl.col("nz_mean")) ** 2)
                  .otherwise(None)
                  .alias("CV2"),
            )
        )

        return stats

    # --------------------------------
    # 기존: 간헐 여부만 (True/False)
    # --------------------------------
    def detect(self, *, return_stats: bool = False) -> pl.DataFrame:
        cfg = self.config
        name_col = cfg.name_col

        stats = self._compute_stats()

        # 최소 히스토리 길이 조건
        enough_history_expr = pl.col("n_periods") >= cfg.min_periods

        cond_adi = pl.col("ADI") >= cfg.adi_threshold

        if cfg.use_cv2:
            cond_cv2 = pl.col("CV2") >= cfg.cv2_threshold
            cond_sparse = (cond_adi | cond_cv2) & enough_history_expr
        else:
            cond_sparse = cond_adi & enough_history_expr

        stats = stats.with_columns(
            pl.when(cond_sparse)
              .then(True)
              .otherwise(False)
              .alias("is_sparsity")
        )

        if return_stats:
            cols = [
                name_col, "is_sparsity",
                "n_periods", "n_zero", "n_nz",
                "zero_ratio", "ADI", "CV2",
            ]
            return stats.select(cols)

        return stats.select([name_col, "is_sparsity"])

    # --------------------------------
    # ① 수요 유형 분류 (smooth / erratic / intermittent / lumpy)
    # --------------------------------
    def classify(self, *, return_stats: bool = False) -> pl.DataFrame:
        """
        ADI / CV² 기준으로 수요 유형을 분류합니다.

        - ADI < adi_th, CV² < cv2_th   → smooth
        - ADI < adi_th, CV² ≥ cv2_th   → erratic
        - ADI ≥ adi_th, CV² < cv2_th   → intermittent
        - ADI ≥ adi_th, CV² ≥ cv2_th   → lumpy
        - (히스토리 부족 or ADI/CV² 계산 불가) → insufficient
        """
        cfg = self.config
        name_col = cfg.name_col

        stats = self._compute_stats()

        # 히스토리 충분 여부 컬럼
        stats = stats.with_columns(
            (pl.col("n_periods") >= cfg.min_periods).alias("_enough_history")
        )

        adi = pl.col("ADI")
        cv2 = pl.col("CV2")
        enough = pl.col("_enough_history")

        # demand_type 컬럼 생성
        stats = stats.with_columns(
            pl.when(
                (~enough) | adi.is_null() | cv2.is_null()
            )
            .then(pl.lit("insufficient"))
            .when((adi < cfg.adi_threshold) & (cv2 < cfg.cv2_threshold))
            .then(pl.lit("smooth"))
            .when((adi < cfg.adi_threshold) & (cv2 >= cfg.cv2_threshold))
            .then(pl.lit("erratic"))
            .when((adi >= cfg.adi_threshold) & (cv2 < cfg.cv2_threshold))
            .then(pl.lit("intermittent"))
            .otherwise(pl.lit("lumpy"))
            .alias("demand_type")
        )

        # 필요하면 간헐 여부도 같이 붙일 수 있음 (intermittent/lumpy → True)
        stats = stats.with_columns(
            pl.col("demand_type")
            .is_in(["intermittent", "lumpy"])
            .alias("is_sparsity")
        )

        if return_stats:
            cols = [
                name_col, "demand_type", "is_sparsity",
                "n_periods", "n_zero", "n_nz",
                "zero_ratio", "ADI", "CV2",
            ]
            return stats.select(cols)

        return stats.select([name_col, "demand_type"])

# weekly_df: 연속 주차로 구성된 주간 수요 테이블
# cfg = IntermittentConfig(
#     name_col="oper_part_no",
#     target_col="demand_qty",
#     adi_threshold=1.32,
#     cv2_threshold=0.49,
#     count_threshold=20,   # 20 미만은 사실상 0으로 취급
#     use_cv2=True,
#     min_periods=20,
# )
#
# detector = IntermittentDemandDetector(weekly_df, config=cfg)
#
# # 1) 간헐 여부 플래그만
# sparsity_flag = detector.detect(return_stats=False)
# # → oper_part_no | is_sparsity
#
# # 2) 수요 유형(smooth / erratic / intermittent / lumpy / insufficient)
# demand_type = detector.classify(return_stats=False)
# # → oper_part_no | demand_type
#
# # 3) 타입 + ADI / CV²까지 같이 보고 싶을 때
# demand_type_detail = detector.classify(return_stats=True)
# # → oper_part_no | demand_type | is_sparsity | n_periods | ... | ADI | CV2