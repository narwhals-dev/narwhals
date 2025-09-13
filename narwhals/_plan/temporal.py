from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from narwhals._duration import Interval
from narwhals._plan.common import ExprNamespace, Function, IRNamespace
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from typing_extensions import TypeAlias, TypeIs

    from narwhals._duration import IntervalUnit
    from narwhals._plan.dummy import Expr
    from narwhals.typing import TimeUnit

PolarsTimeUnit: TypeAlias = Literal["ns", "us", "ms"]


def _is_polars_time_unit(obj: Any) -> TypeIs[PolarsTimeUnit]:
    return obj in {"ns", "us", "ms"}


# fmt: off
class TemporalFunction(Function, accessor="dt", options=FunctionOptions.elementwise): ...
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
# fmt: on
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
    time_unit: PolarsTimeUnit

    @staticmethod
    def from_time_unit(time_unit: TimeUnit = "us", /) -> Timestamp:
        if not _is_polars_time_unit(time_unit):
            msg = f"invalid `time_unit` \n\nExpected one of ['ns', 'us', 'ms'], got {time_unit!r}."
            raise ValueError(msg)
        return Timestamp(time_unit=time_unit)

    def __repr__(self) -> str:
        return f"{super().__repr__()}[{self.time_unit!r}]"


class Truncate(TemporalFunction):
    __slots__ = ("multiple", "unit")
    multiple: int
    unit: IntervalUnit

    @staticmethod
    def from_string(every: str, /) -> Truncate:
        return Truncate.from_interval(Interval.parse(every))

    @staticmethod
    def from_interval(every: Interval, /) -> Truncate:
        return Truncate(multiple=every.multiple, unit=every.unit)


class IRDateTimeNamespace(IRNamespace):
    date: ClassVar = Date
    year: ClassVar = Year
    month: ClassVar = Month
    day: ClassVar = Day
    hour: ClassVar = Hour
    minute: ClassVar = Minute
    second: ClassVar = Second
    millisecond: ClassVar = Millisecond
    microsecond: ClassVar = Microsecond
    nanosecond: ClassVar = Nanosecond
    ordinal_day: ClassVar = OrdinalDay
    weekday: ClassVar = WeekDay
    total_minutes: ClassVar = TotalMinutes
    total_seconds: ClassVar = TotalSeconds
    total_milliseconds: ClassVar = TotalMilliseconds
    total_microseconds: ClassVar = TotalMicroseconds
    total_nanoseconds: ClassVar = TotalNanoseconds
    to_string: ClassVar = ToString
    replace_time_zone: ClassVar = ReplaceTimeZone
    convert_time_zone: ClassVar = ConvertTimeZone
    truncate: ClassVar = staticmethod(Truncate.from_string)
    timestamp: ClassVar = staticmethod(Timestamp.from_time_unit)


class ExprDateTimeNamespace(ExprNamespace[IRDateTimeNamespace]):
    @property
    def _ir_namespace(self) -> type[IRDateTimeNamespace]:
        return IRDateTimeNamespace

    def date(self) -> Expr:
        return self._with_unary(self._ir.date())

    def year(self) -> Expr:
        return self._with_unary(self._ir.year())

    def month(self) -> Expr:
        return self._with_unary(self._ir.month())

    def day(self) -> Expr:
        return self._with_unary(self._ir.day())

    def hour(self) -> Expr:
        return self._with_unary(self._ir.hour())

    def minute(self) -> Expr:
        return self._with_unary(self._ir.minute())

    def second(self) -> Expr:
        return self._with_unary(self._ir.second())

    def millisecond(self) -> Expr:
        return self._with_unary(self._ir.millisecond())

    def microsecond(self) -> Expr:
        return self._with_unary(self._ir.microsecond())

    def nanosecond(self) -> Expr:
        return self._with_unary(self._ir.nanosecond())

    def ordinal_day(self) -> Expr:
        return self._with_unary(self._ir.ordinal_day())

    def weekday(self) -> Expr:
        return self._with_unary(self._ir.weekday())

    def total_minutes(self) -> Expr:
        return self._with_unary(self._ir.total_minutes())

    def total_seconds(self) -> Expr:
        return self._with_unary(self._ir.total_seconds())

    def total_milliseconds(self) -> Expr:
        return self._with_unary(self._ir.total_milliseconds())

    def total_microseconds(self) -> Expr:
        return self._with_unary(self._ir.total_microseconds())

    def total_nanoseconds(self) -> Expr:
        return self._with_unary(self._ir.total_nanoseconds())

    def to_string(self, format: str) -> Expr:
        return self._with_unary(self._ir.to_string(format=format))

    def replace_time_zone(self, time_zone: str | None) -> Expr:
        return self._with_unary(self._ir.replace_time_zone(time_zone=time_zone))

    def convert_time_zone(self, time_zone: str) -> Expr:
        return self._with_unary(self._ir.convert_time_zone(time_zone=time_zone))

    def timestamp(self, time_unit: TimeUnit = "us") -> Expr:
        return self._with_unary(self._ir.timestamp(time_unit))

    def truncate(self, every: str) -> Expr:
        return self._with_unary(self._ir.truncate(every))
