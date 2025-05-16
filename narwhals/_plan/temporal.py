from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import Function
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from narwhals.typing import TimeUnit


class TemporalFunction(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


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
