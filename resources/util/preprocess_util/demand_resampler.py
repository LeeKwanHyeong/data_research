from typing import Literal
import polars as pl


class DemandResampler:
    """
    일(日) 단위 수요 데이터(yyyymmdd Int)를
    - 주(週) 단위 (ISO 기준 월요일 시작 주)
    - 월(月) 단위
    로 집계하고, 각 품목별로 min~max 구간 사이의 누락된 기간을 0으로 채워주는 헬퍼 클래스.

    Weekly:
      - 주 시작일: 월요일(Date)
      - ISO 주차: iso_yyyyww (예: 202401, 202452 등, ISO year + ISO week)

    Monthly:
      - 월 시작일: 해당 달의 1일(Date)
      - 월 번호: yyyymm (예: 202401, 202412)
    """

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        name_col: str = "oper_part_no",
        date_col: str = "demand_dt",
        target_col: str = "demand_qty",
        date_fmt: str = "%Y%m%d",
    ):
        """
        Parameters
        ----------
        df : pl.DataFrame
            원본 데이터프레임 (일 단위).
            예) ["oper_part_no", "demand_dt"(Int yyyymmdd), "demand_qty"]
        name_col : str, default "oper_part_no"
            부품/아이템 식별자 컬럼명.
        date_col : str, default "demand_dt"
            날짜 컬럼명 (Int yyyymmdd 형태라고 가정).
        target_col : str, default "demand_qty"
            수요량 컬럼명.
        date_fmt : str, default "%Y%m%d"
            date_col을 Date로 파싱할 때 사용할 포맷.
        """
        self.df = df
        self.name_col = name_col
        self.date_col = date_col
        self.target_col = target_col
        self.date_fmt = date_fmt

        # 컬럼 존재 여부 체크
        missing = [c for c in (name_col, date_col, target_col) if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame에 {missing} 컬럼이 없습니다: {missing}")

    # -------------------------
    # 내부 공통 유틸
    # -------------------------
    def _with_date(self) -> pl.DataFrame:
        """
        self.date_col(Int yyyymmdd) → Date 컬럼 "_date" 추가
        """
        return self.df.with_columns(
            pl.col(self.date_col)
            .cast(pl.Utf8)
            .str.strptime(pl.Date, format=self.date_fmt)
            .alias("_date")
        )

    @staticmethod
    def _build_calendar(
        grouped: pl.DataFrame,
        name_col: str,
        date_col: str,
        interval: str,
    ) -> pl.DataFrame:
        """
        주간/월간 공통: 각 name별 min~max 날짜로부터 연속 calendar 생성

        grouped : 이미 [name_col, date_col] 기준으로 집계된 테이블
        interval: "1w" 또는 "1mo"
        """
        part_minmax = (
            grouped
            .group_by(name_col)
            .agg(
                pl.col(date_col).min().alias("min_date"),
                pl.col(date_col).max().alias("max_date"),
            )
        )

        calendar = (
            part_minmax
            .with_columns(
                pl.date_ranges(
                    start=pl.col("min_date"),
                    end=pl.col("max_date"),
                    interval=interval,
                    closed="both",
                ).alias(date_col)
            )
            .explode(date_col)
            .select([name_col, date_col])
        )
        return calendar

    # -------------------------
    # PUBLIC: 주간 채움
    # -------------------------
    def to_weekly_filled(
        self,
        *,
        as_int: bool = False,
        add_iso_yyyyww: bool = True,
    ) -> pl.DataFrame:
        """
        일 단위 데이터를 주(週) 단위로 집계 후, 각 품목별 min_week~max_week 사이의
        누락 주차를 0으로 채워 반환.

        - 주 시작일은 ISO와 동일하게 "월요일"을 기준으로 truncate.
        - 옵션으로 ISO yyyyww(Int) 컬럼을 추가할 수 있음.

        Parameters
        ----------
        as_int : bool, default False
            True이면 주 시작일을 Int(yyyymmdd)로 변환해서 반환.
        add_iso_yyyyww : bool, default True
            True이면 iso_yyyyww(Int, ISO year + ISO week) 컬럼을 추가.

        Returns
        -------
        pl.DataFrame
            - self.date_col (Date or Int): 주 시작일 (월요일)
            - self.target_col : 주간 합계 수요량, 비어 있던 주차는 0
            - iso_yyyyww (선택): ISO 기준 yyyyww(Int)
        """
        name_col = self.name_col
        date_col = self.date_col
        target_col = self.target_col

        df_date = self._with_date()

        # 1) 주 시작일로 truncate 후 주간 집계 (월요일 anchor)
        df_weekly = (
            df_date
            .with_columns(
                pl.col("_date")
                .dt.truncate("1w")  # 월요일 기준 주 시작일
                .alias("week_start")
            )
            .group_by([name_col, "week_start"])
            .agg(
                pl.col(target_col).sum().alias("weekly_qty")
            )
        )

        # 2) calendar 생성 (연속된 week_start)
        calendar = self._build_calendar(
            grouped=df_weekly,
            name_col=name_col,
            date_col="week_start",
            interval="1w",
        )

        # 3) left join + null → 0
        weekly_filled = (
            calendar
            .join(df_weekly, on=[name_col, "week_start"], how="left")
            .with_columns(
                pl.col("weekly_qty").fill_null(0)
            )
            .sort([name_col, "week_start"])
        )

        # 4) ISO yyyyww 컬럼 추가 (ISO year + ISO week)
        #   - %G : ISO year
        #   - %V : ISO week number (01-53)
        if add_iso_yyyyww:
            weekly_filled = weekly_filled.with_columns(
                pl.col("week_start")
                .dt.strftime("%G%V")    # 문자열 "202401" 등
                .cast(pl.Int64)
                .alias("iso_yyyyww")
            )

        # 5) date_col / target_col 이름 정리
        weekly_filled = weekly_filled.rename(
            {
                "week_start": date_col,
                "weekly_qty": target_col,
            }
        )

        # 6) 날짜를 다시 Int로 쓰고 싶을 때
        if as_int:
            weekly_filled = weekly_filled.with_columns(
                pl.col(date_col).dt.strftime(self.date_fmt).cast(pl.Int64)
            )

        return weekly_filled

    # -------------------------
    # PUBLIC: 월간 채움
    # -------------------------
    def to_monthly_filled(
        self,
        *,
        as_int: bool = False,
        add_yyyymm: bool = True,
    ) -> pl.DataFrame:
        """
        일 단위 데이터를 월(月) 단위로 집계 후, 각 품목별 min_month~max_month 사이의
        누락 월을 0으로 채워 반환.

        - 월 시작일은 해당 달의 1일로 truncate.
        - 옵션으로 yyyymm(Int) 컬럼을 추가할 수 있음.

        Parameters
        ----------
        as_int : bool, default False
            True이면 월 시작일(1일)을 Int(yyyymmdd)로 변환해서 반환.
        add_yyyymm : bool, default True
            True이면 yyyymm(Int) 컬럼을 추가.

        Returns
        -------
        pl.DataFrame
            - self.date_col (Date or Int): 월 시작일(1일)
            - self.target_col : 월간 합계 수요량, 비어 있던 월은 0
            - yyyymm (선택): 월 번호(Int)
        """
        name_col = self.name_col
        date_col = self.date_col
        target_col = self.target_col

        df_date = self._with_date()

        # 1) 월 시작일로 truncate 후 월간 집계 (해당 달 1일)
        df_monthly = (
            df_date
            .with_columns(
                pl.col("_date")
                .dt.truncate("1mo")  # 해당 달의 1일
                .alias("month_start")
            )
            .group_by([name_col, "month_start"])
            .agg(
                pl.col(target_col).sum().alias("monthly_qty")
            )
        )

        # 2) calendar 생성 (연속된 month_start)
        calendar = self._build_calendar(
            grouped=df_monthly,
            name_col=name_col,
            date_col="month_start",
            interval="1mo",
        )

        # 3) left join + null → 0
        monthly_filled = (
            calendar
            .join(df_monthly, on=[name_col, "month_start"], how="left")
            .with_columns(
                pl.col("monthly_qty").fill_null(0)
            )
            .sort([name_col, "month_start"])
        )

        # 4) yyyymm 컬럼 추가
        if add_yyyymm:
            monthly_filled = monthly_filled.with_columns(
                pl.col("month_start")
                .dt.strftime("%Y%m")
                .cast(pl.Int64)
                .alias("yyyymm")
            )

        # 5) date_col / target_col 이름 정리
        monthly_filled = monthly_filled.rename(
            {
                "month_start": date_col,
                "monthly_qty": target_col,
            }
        )

        # 6) 날짜를 다시 Int(yyyymmdd)로 쓰고 싶을 때
        if as_int:
            monthly_filled = monthly_filled.with_columns(
                pl.col(date_col).dt.strftime(self.date_fmt).cast(pl.Int64)
            )

        return monthly_filled

    # -------------------------
    # 선택형 인터페이스
    # -------------------------
    def fill(
        self,
        freq: Literal["weekly", "monthly"],
        *,
        as_int: bool = False,
        add_iso_yyyyww: bool = True,
        add_yyyymm: bool = True,
    ) -> pl.DataFrame:
        """
        freq에 따라 weekly / monthly 중 선택해서 반환.

        Parameters
        ----------
        freq : {"weekly", "monthly"}
        as_int : bool, default False
            True이면 날짜를 Int(yyyymmdd)로 변환.
        add_iso_yyyyww : bool, default True
            weekly일 때, iso_yyyyww(Int) 컬럼 추가 여부.
        add_yyyymm : bool, default True
            monthly일 때, yyyymm(Int) 컬럼 추가 여부.

        Returns
        -------
        pl.DataFrame
        """
        if freq == "weekly":
            return self.to_weekly_filled(
                as_int=as_int,
                add_iso_yyyyww=add_iso_yyyyww,
            )
        elif freq == "monthly":
            return self.to_monthly_filled(
                as_int=as_int,
                add_yyyymm=add_yyyymm,
            )
        else:
            raise ValueError("freq는 'weekly' 또는 'monthly'만 허용됩니다.")


# -------------------------
# 사용 예시
# -------------------------
# resampler = DemandResampler(
#     target_dyn_demand,
#     name_col="oper_part_no",
#     date_col="demand_dt",
#     target_col="demand_qty",
#     date_fmt="%Y%m%d",
# )
#
# # 1) 주간: Date + iso_yyyyww
# weekly_df = resampler.to_weekly_filled(as_int=False, add_iso_yyyyww=True)
#
# # 2) 월간: Date + yyyymm
# monthly_df = resampler.to_monthly_filled(as_int=False, add_yyyymm=True)
#
# # 3) 선택형 인터페이스
# weekly_df2 = resampler.fill("weekly", as_int=True, add_iso_yyyyww=True)
# monthly_df2 = resampler.fill("monthly", as_int=True, add_yyyymm=True)
