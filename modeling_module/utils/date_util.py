# ==== DateUtil: YYYYMM 전용 유틸 추가 ====
from datetime import date, datetime, timedelta
from typing import List, Iterable

import numpy as np
import pandas as pd
import polars as pl

class DateUtil:
    @staticmethod
    def yyyymmdd_to_date(yyyymmdd: int) -> date:
        return datetime.strptime(str(yyyymmdd), '%Y%m%d').date()

    @staticmethod
    def add_months_to_date(dt: datetime, months: int) -> datetime:
        year = dt.year + (dt.month + months - 1) // 12
        month = (dt.month + months - 1) % 12 + 1
        day = min(dt.day, [31,
                           29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                           31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
        return datetime(year, month, day)

    @staticmethod
    def datetime_to_yyyymmdd(dt: datetime) -> int:
        return int(dt.strftime('%Y%m%d'))

    @staticmethod
    def yyyyww_to_date(yyyyww: int) -> date:
        yyyy = yyyyww // 100
        ww = yyyyww % 100
        return date.fromisocalendar(yyyy, ww, 1)

    @staticmethod
    def yyyymm_to_date(yyyymm: int):
        return datetime.strptime(str(yyyymm), '%Y%m').date()

    @staticmethod
    def date_to_yyyyww(dt: date) -> int:
        iso = dt.isocalendar()
        return iso[0] * 100 + iso[1]

    @staticmethod
    def date_to_yyyymm(dt: date) -> int:
        return int(dt.strftime('%Y%m'))

    @staticmethod
    def extend_weeks_proper(df: pl.DataFrame) -> pl.DataFrame:
        oper_part = df[0, 'oper_part_no']
        weeks_to_add = df.shape[0]
        last_yyyyww = df['order_yyyyww'].max()
        last_date = DateUtil.yyyyww_to_date(last_yyyyww)

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
    def add_weeks_yyyyww(yyyyww: int, delta_weeks: int) -> int:
        year = int(yyyyww) // 100
        week = int(yyyyww) % 100
        if week < 1 or week > 53:
            raise ValueError(f"Invalid ISO week in yyyyww={yyyyww} (week must be 1..53)")
        try:
            d = date.fromisocalendar(year, week, 1)  # Monday of that ISO week
        except ValueError as e:
            raise ValueError(f"Invalid yyyyww={yyyyww}: {e}")
        d2 = d + timedelta(weeks=int(delta_weeks))
        iso = d2.isocalendar()
        return int(iso[0]) * 100 + int(iso[1])

    @staticmethod
    def week_seq_ending_before(plan_yyyyww: int, lookback: int) -> List[int]:
        """
        plan_yyyyww '직전' 주부터 과거로 거슬러 올라가며 lookback개 주차(YYYYWW)를 반환.
        반환은 오래된→최근 순 (모델 입력 히스토리와 동일한 시간 진행 방향).
        예) plan=202452, lookback=3  => [202449, 202450, 202451]
        """
        if lookback <= 0:
            return []
        # plan 직전 주부터 시작
        cur = DateUtil.add_weeks_yyyyww(plan_yyyyww, -1)
        seq = [cur]
        for _ in range(lookback - 1):
            cur = DateUtil.add_weeks_yyyyww(cur, -1)
            seq.append(cur)
        seq.reverse()  # 오래된→최근
        return seq

    @classmethod
    def next_n_weeks_from(cls,
                          plan_yyyyww: int,
                          n: int,
                          include_anchor: bool = False) -> List[int]:
        """
        plan_yyyyww를 기준으로 미래 주차 시퀀스(길이 n)를 반환.
        - include_anchor=False: plan+1주, plan+2주, ... 로 H개
        - include_anchor=True : plan주, plan+1주, ... 로 H개
        """
        if n <= 0:
            return []
        cur = plan_yyyyww if include_anchor else cls.add_weeks_yyyyww(plan_yyyyww, +1)
        seq = [cur]
        for _ in range(n - 1):
            cur = cls.add_weeks_yyyyww(cur, +1)
            seq.append(cur)
        return seq

    @staticmethod
    def yyyyww_to_datetime(weeks: Iterable[int] | int):
        """
        YYYYWW(ISO week) -> 해당 주의 '월요일' datetime으로 변환.
        - 입력이 스칼라면 단일 datetime.date 반환
        - 입력이 리스트/배열이면 numpy datetime64[D] 배열 반환
        """

        def _one(wk: int) -> date:
            y = int(wk) // 100
            w = int(wk) % 100
            if w < 1 or w > 53:
                raise ValueError(f"Invalid ISO week in yyyyww={wk} (week must be 1..53)")
            try:
                return date.fromisocalendar(y, w, 1)  # Monday
            except ValueError as e:
                raise ValueError(f"Invalid yyyyww={wk}: {e}")

        if isinstance(weeks, (int, np.integer)):
            return _one(int(weeks))

        # iterable
        dates = [_one(int(w)) for w in weeks]
        # numpy datetime64[D]로 반환하면 matplotlib 등과 궁합이 좋습니다.
        return np.array(dates, dtype='datetime64[D]')

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
# ---------- 문자열 날짜 파싱 ----------
    @staticmethod
    def parse_to_yyyymmdd(date_str: str) -> int:
        """
        '2016-07-01 00:00:00' 또는 '2016-07-01' 같은 문자열을
        int형 YYYYMMDD (20160701)로 변환
        """
        # 공백이나 시간 부분 제거 후 날짜 부분만 추출
        s = str(date_str).strip().split(' ')[0]  # '2016-07-01'
        # 구분자 제거
        for sep in ['-', '.', '/']:
            s = s.replace(sep, '')
        return int(s)

    @staticmethod
    def parse_to_yyyymmddhh(date_str: str) -> int:
        """
        '2016-07-01 12:30:00' -> int YYYYMMDDHH (2016070112)
        """
        # datetime 객체로 파싱 시도 (다양한 포맷 대응)
        try:
            # 일반적인 포맷 시도
            dt = pd.to_datetime(date_str)
            return int(dt.strftime('%Y%m%d%H'))
        except Exception:
            # 수동 파싱 (fallback)
            s = str(date_str).strip()
            # 숫자만 남김
            for sep in ['-', '.', '/', ':', ' ']:
                s = s.replace(sep, '')
            # YYYYMMDDHH (10자리) 까지만 자름
            return int(s[:10])

    # ---------- 일간/시간 연산 ----------
    @staticmethod
    def add_days_yyyymmdd(yyyymmdd: int, days: int) -> int:
        """YYYYMMDD 정수에 일수 더하기/빼기"""
        s = str(yyyymmdd)
        dt = datetime.strptime(s, "%Y%m%d")
        new_dt = dt + timedelta(days=days)
        return int(new_dt.strftime("%Y%m%d"))

    @staticmethod
    def add_hours_yyyymmddhh(yyyymmddhh: int, hours: int) -> int:
        """YYYYMMDDHH 정수에 시간 더하기/빼기"""
        s = str(yyyymmddhh)
        dt = datetime.strptime(s, "%Y%m%d%H")
        new_dt = dt + timedelta(hours=hours)
        return int(new_dt.strftime("%Y%m%d%H"))