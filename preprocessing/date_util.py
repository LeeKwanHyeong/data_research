import datetime
from datetime import datetime
from datetime import date, timedelta
import polars as pl

class DateUtil:
    def __int__(self):
        pass

    def yyyymmdd_to_date(self, yyyymmdd: int) -> datetime:
        return datetime.strptime(str(yyyymmdd), '%Y%m%d')

    def add_months_to_date(self, date: datetime, months: int) -> datetime:
        year = date.year + (date.month + months - 1) // 12
        month = (date.month + months - 1) % 12 + 1
        return datetime(year, month, date.day)

    def datetime_to_yyyymmdd(self, date: datetime) -> int:
        return int(date.strftime('%Y%m%d'))

    def yyyyww_to_date(self, yyyyww: int):
        yyyy = yyyyww // 100
        ww = yyyyww % 100
        # ISO 주차 시작일은 그 주의 월요일
        return datetime.date.fromisocalendar(yyyy, ww, 1)

    def date_to_yyyyww(self, dt: date) -> int:
        iso = dt.isocalendar()
        return iso[0] * 100 + iso[1]

    def extend_weeks_proper(self, df: pl.DataFrame) -> pl.DataFrame:
        oper_part = df[0, 'oper_part_no']
        weeks_to_add = df.shape[0]
        last_yyyyww = df['order_yyyyww'].max()
        last_date = self.yyyyww_to_date(last_yyyyww)

        extended_dates = [last_date + timedelta(weeks = i) for i in range(1, weeks_to_add + 1)]
        extended_yyyywws = [self.date_to_yyyyww(d) for d in extended_dates]

        return pl.DataFrame({'oper_part_no': [oper_part] * weeks_to_add,
                             'order_yyyyww': extended_yyyywws,
                             'order_qty': [0] * weeks_to_add})

