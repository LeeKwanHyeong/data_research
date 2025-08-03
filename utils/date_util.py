import datetime
from datetime import datetime
from datetime import date, timedelta
import polars as pl

# =========================
# 1. DateUtil 클래스
# =========================
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

        extended_dates = [last_date + timedelta(weeks=i) for i in range(1, weeks_to_add + 1)]
        extended_yyyywws = [DateUtil.date_to_yyyyww(d) for d in extended_dates]

        return pl.DataFrame({
            'oper_part_no': [oper_part] * weeks_to_add,
            'order_yyyyww': extended_yyyywws,
            'order_qty': [0] * weeks_to_add
        })