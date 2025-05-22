from __future__ import annotations

from typing import TYPE_CHECKING, cast

from narwhals._plan.common import Function
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from narwhals.typing import TimeUnit


class TemporalFunction(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        tp = type(self)
        if tp is TemporalFunction:
            return tp.__name__
        elif tp is Timestamp:
            tu = cast("Timestamp", self).time_unit
            return f"dt.timestamp[{tu!r}]"
        m: dict[type[TemporalFunction], str] = {
            Year: "year",
            Month: "month",
            WeekDay: "weekday",
            Day: "day",
            OrdinalDay: "ordinal_day",
            Date: "date",
            Hour: "hour",
            Minute: "minute",
            Second: "second",
            Millisecond: "millisecond",
            Microsecond: "microsecond",
            Nanosecond: "nanosecond",
            TotalMinutes: "total_minutes",
            TotalSeconds: "total_seconds",
            TotalMilliseconds: "total_milliseconds",
            TotalMicroseconds: "total_microseconds",
            TotalNanoseconds: "total_nanoseconds",
            ToString: "to_string",
            ConvertTimeZone: "convert_time_zone",
            ReplaceTimeZone: "replace_time_zone",
            Truncate: "truncate",
        }
        return f"dt.{m[tp]}"


class Date(TemporalFunction): ...


class Year(TemporalFunction): ...


class Month(TemporalFunction): ...


class Day(TemporalFunction): ...


class Hour(TemporalFunction): ...


class Minute(TemporalFunction): ...


class Second(TemporalFunction): ...


class Millisecond(TemporalFunction): ...


class Microsecond(TemporalFunction): ...


class Nanosecond(TemporalFunction): ...


class OrdinalDay(TemporalFunction): ...


class WeekDay(TemporalFunction): ...


class TotalMinutes(TemporalFunction): ...


class TotalSeconds(TemporalFunction): ...


class TotalMilliseconds(TemporalFunction): ...


class TotalMicroseconds(TemporalFunction): ...


class TotalNanoseconds(TemporalFunction): ...


class ToString(TemporalFunction):
    __slots__ = ("format",)

    format: str


class ReplaceTimeZone(TemporalFunction):
    __slots__ = ("time_zone",)

    time_zone: str | None


class ConvertTimeZone(TemporalFunction):
    __slots__ = ("time_zone",)

    time_zone: str


class Timestamp(TemporalFunction):
    __slots__ = ("time_unit",)

    time_unit: TimeUnit


class Truncate(TemporalFunction):
    __slots__ = ("every",)

    every: str
