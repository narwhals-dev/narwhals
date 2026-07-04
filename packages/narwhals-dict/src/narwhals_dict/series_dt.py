from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from narwhals._compliant import EagerSeriesNamespace
from narwhals._compliant.any_namespace import DateTimeNamespace
from narwhals._duration import Interval
from narwhals_dict.utils import (
    EPOCH_NAIVE,
    MICROSECONDS_PER_UNIT,
    datetime_to_us,
    timedelta_to_us,
    trunc_div,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from narwhals.typing import TimeUnit
    from narwhals_dict.series import DictSeries

_US_PER_MINUTE = 60_000_000


def _add_months(value: datetime | date, months: int) -> datetime | date:
    month0 = value.month - 1 + months
    year = value.year + month0 // 12
    month = month0 % 12 + 1
    day = min(value.day, monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


class DictSeriesDateTimeNamespace(
    EagerSeriesNamespace["DictSeries", Any], DateTimeNamespace["DictSeries"]
):
    def _unary(self, fn: Callable[[Any], Any]) -> DictSeries:
        return self.with_native(
            [None if value is None else fn(value) for value in self.native]
        )

    def to_string(self, format: str) -> DictSeries:
        # Polars' parser treats `'%.f'` as `strftime` does `'.%f'`.
        fmt = format.replace("%S%.f", "%S.%f")
        return self._unary(lambda value: value.strftime(fmt))

    def replace_time_zone(self, time_zone: str | None) -> DictSeries:
        from zoneinfo import ZoneInfo

        tzinfo = ZoneInfo(time_zone) if time_zone is not None else None
        return self._unary(lambda value: value.replace(tzinfo=tzinfo))

    def convert_time_zone(self, time_zone: str) -> DictSeries:
        from zoneinfo import ZoneInfo

        tzinfo = ZoneInfo(time_zone)

        def fn(value: datetime) -> datetime:
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(tzinfo)

        return self._unary(fn)

    def timestamp(self, time_unit: TimeUnit) -> DictSeries:
        dtypes = self.version.dtypes
        dtype = self.compliant.dtype
        if not isinstance(dtype, (dtypes.Datetime, dtypes.Date)):
            msg = "Input should be either of Date or Datetime type."
            raise TypeError(msg)

        def fn(value: Any) -> int:
            us = (
                datetime_to_us(value)
                if isinstance(value, datetime)
                else (value - date(1970, 1, 1)).days * 86_400_000_000
            )
            if time_unit == "ns":
                return us * 1_000
            return trunc_div(us, MICROSECONDS_PER_UNIT[time_unit])

        return self._unary(fn)

    def date(self) -> DictSeries:
        return self._unary(lambda value: value.date())

    def year(self) -> DictSeries:
        return self._unary(lambda value: value.year)

    def month(self) -> DictSeries:
        return self._unary(lambda value: value.month)

    def day(self) -> DictSeries:
        return self._unary(lambda value: value.day)

    def hour(self) -> DictSeries:
        return self._unary(lambda value: value.hour)

    def minute(self) -> DictSeries:
        return self._unary(lambda value: value.minute)

    def second(self) -> DictSeries:
        return self._unary(lambda value: value.second)

    def millisecond(self) -> DictSeries:
        return self._unary(lambda value: value.microsecond // 1_000)

    def microsecond(self) -> DictSeries:
        return self._unary(lambda value: value.microsecond)

    def nanosecond(self) -> DictSeries:
        return self._unary(lambda value: value.microsecond * 1_000)

    def ordinal_day(self) -> DictSeries:
        return self._unary(lambda value: value.timetuple().tm_yday)

    def weekday(self) -> DictSeries:
        return self._unary(lambda value: value.isoweekday())

    def _total(self, us_per_unit: int) -> DictSeries:
        return self._unary(lambda value: trunc_div(timedelta_to_us(value), us_per_unit))

    def total_minutes(self) -> DictSeries:
        return self._total(_US_PER_MINUTE)

    def total_seconds(self) -> DictSeries:
        return self._total(1_000_000)

    def total_milliseconds(self) -> DictSeries:
        return self._total(1_000)

    def total_microseconds(self) -> DictSeries:
        return self._total(1)

    def total_nanoseconds(self) -> DictSeries:
        return self._unary(lambda value: timedelta_to_us(value) * 1_000)

    def truncate(self, every: str) -> DictSeries:
        interval = Interval.parse(every)
        multiple, unit = interval.multiple, interval.unit

        if unit in {"mo", "q", "y"}:
            months = multiple * {"mo": 1, "q": 3, "y": 12}[unit]

            def fn(value: datetime | date) -> datetime | date:
                total_months = (value.year - 1970) * 12 + value.month - 1
                floored = total_months - total_months % months
                start = value.replace(
                    year=1970 + floored // 12, month=floored % 12 + 1, day=1
                )
                if isinstance(start, datetime):
                    return start.replace(hour=0, minute=0, second=0, microsecond=0)
                return start

            return self._unary(fn)

        if unit == "ns":
            if multiple % 1_000 == 0:
                multiple, unit = multiple // 1_000, "us"
            elif 1_000 % multiple == 0:
                # Microsecond-precision values are already aligned: no-op.
                return self._unary(lambda value: value)
            else:  # pragma: no cover
                msg = "Truncating to 'ns' is not supported for the dict backend."
                raise NotImplementedError(msg)

        every_us = timedelta_to_us(Interval(multiple, unit).to_timedelta())

        def fn_time(value: datetime | date) -> datetime | date:
            if not isinstance(value, datetime):
                value = datetime(value.year, value.month, value.day)
            tzinfo = value.tzinfo
            # Truncate in local wall time, like Polars.
            naive = value.replace(tzinfo=None)
            us = datetime_to_us(naive)
            truncated = EPOCH_NAIVE + timedelta(microseconds=us - us % every_us)
            return truncated.replace(tzinfo=tzinfo)

        return self._unary(fn_time)

    def offset_by(self, by: str) -> DictSeries:
        interval = Interval.parse_no_constraints(by)
        multiple, unit = interval.multiple, interval.unit

        if unit in {"mo", "q", "y"}:
            months = multiple * {"mo": 1, "q": 3, "y": 12}[unit]
            return self._unary(lambda value: _add_months(value, months))

        if unit == "ns":
            if multiple % 1_000:  # pragma: no cover
                msg = "Sub-microsecond offsets are not supported for the dict backend."
                raise NotImplementedError(msg)
            delta = timedelta(microseconds=multiple // 1_000)
        else:
            delta = Interval(multiple, unit).to_timedelta()

        return self._unary(lambda value: value + delta)
