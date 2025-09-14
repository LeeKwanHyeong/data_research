# ==== DateUtil: YYYYMM 전용 유틸 추가 ====
from datetime import date
import numpy as np
import pandas as pd

class DateUtil:
    # ... (기존 메서드들은 그대로 두세요)

    # ---------- 기본 파싱/검증 ----------
    @staticmethod
    def parse_yyyymm(x) -> int:
        """
        int(202210) | str('202210'/'2022-10'/'2022.10'/'2022/10') → int YYYYMM
        """
        if isinstance(x, int):
            yyyymm = x
        else:
            s = str(x).strip()
            for sep in ['.', '-', '/']:
                s = s.replace(sep, '')
            if len(s) != 6 or not s.isdigit():
                raise ValueError(f"Invalid YYYYMM: {x}")
            yyyymm = int(s)
        if not DateUtil.is_valid_yyyymm(yyyymm):
            raise ValueError(f"Invalid month in YYYYMM: {yyyymm}")
        return yyyymm

    @staticmethod
    def is_valid_yyyymm(yyyymm: int) -> bool:
        y = yyyymm // 100
        m = yyyymm % 100
        return (y >= 1) and (1 <= m <= 12)

    # ---------- 변환 ----------
    @staticmethod
    def yyyymm_to_year_month(yyyymm: int) -> tuple[int, int]:
        yyyymm = DateUtil.parse_yyyymm(yyyymm)
        return yyyymm // 100, yyyymm % 100

    @staticmethod
    def yyyymm_to_date_first(yyyymm: int) -> date:
        """YYYYMM → 그 달 1일(date)"""
        y, m = DateUtil.yyyymm_to_year_month(yyyymm)
        return date(year=y, month=m, day=1)

    @staticmethod
    def date_to_yyyymm(dt: date) -> int:
        return int(dt.strftime("%Y%m"))

    # ---------- 월 가감/차이 ----------
    @staticmethod
    def add_months_yyyymm(yyyymm: int, k: int) -> int:
        """YYYYMM에 k개월 더하기(음수 가능)"""
        y, m = DateUtil.yyyymm_to_year_month(yyyymm)
        m0 = (m - 1) + int(k)
        y2 = y + m0 // 12
        m2 = (m0 % 12) + 1
        return y2 * 100 + m2

    @staticmethod
    def months_between_yyyymm(start: int, end: int, inclusive: bool = False) -> int:
        """
        start→end까지의 '개월 수'.
        inclusive=False: (start, end] 구간 개월 수 (start 다음달부터 end까지)
        inclusive=True : [start, end] 구간 개월 수 (start 포함)
        """
        ys, ms = DateUtil.yyyymm_to_year_month(start)
        ye, me = DateUtil.yyyymm_to_year_month(end)
        diff = (ye - ys) * 12 + (me - ms)
        return diff + (1 if inclusive else 0)

    # ---------- 시퀀스 만들기 ----------
    @staticmethod
    def range_yyyymm(start: int, n: int, include_start: bool = True) -> list[int]:
        """
        시작 월부터 n개 연속 월 시퀀스.
        include_start=True  → [start, start+1, ...] n개
        include_start=False → [start+1, start+2, ...] n개
        """
        start = DateUtil.parse_yyyymm(start)
        base = start if include_start else DateUtil.add_months_yyyymm(start, 1)
        return [DateUtil.add_months_yyyymm(base, i) for i in range(n)]

    @staticmethod
    def month_seq_ending_before(anchor: int, lookback: int) -> list[int]:
        """
        앵커 직전까지의 연속 lookback개월 (예: anchor=202210, lookback=6 → [202204..202209])
        """
        anchor = DateUtil.parse_yyyymm(anchor)
        last = DateUtil.add_months_yyyymm(anchor, -1)
        first = DateUtil.add_months_yyyymm(last, -(lookback - 1))
        return DateUtil.range_yyyymm(first, lookback, include_start=True)

    @staticmethod
    def next_n_months_from(anchor: int, n: int, include_anchor: bool = True) -> list[int]:
        """
        앵커부터 n개월 미래 달력.
        include_anchor=True  → [anchor, anchor+1, ...]
        include_anchor=False → [anchor+1, anchor+2, ...]
        """
        return DateUtil.range_yyyymm(anchor, n, include_start=include_anchor)

    # ---------- 포매팅 ----------
    @staticmethod
    def yyyymm_to_str(yyyymm: int, sep: str = "") -> str:
        """202210 → '202210' or '2022{sep}10'"""
        y, m = DateUtil.yyyymm_to_year_month(yyyymm)
        if sep:
            return f"{y}{sep}{m:02d}"
        return f"{y}{m:02d}"
    @staticmethod
    def yyyymm_to_datetime(arr_like):
        """
        [YYYYMM, ...] → month-start DatetimeIndex
        """
        arr = np.asarray(arr_like, dtype=np.int64)
        s = pd.Series(arr.astype(str)) + "01"  # YYYYMM01
        return pd.to_datetime(s, format="%Y%m%d")
