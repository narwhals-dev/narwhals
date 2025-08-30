from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

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
    def from_time_unit(time_unit: TimeUnit, /) -> Timestamp:
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
    def date(self) -> Date:
        return Date()

    def year(self) -> Year:
        return Year()

    def month(self) -> Month:
        return Month()

    def day(self) -> Day:
        return Day()

    def hour(self) -> Hour:
        return Hour()

    def minute(self) -> Minute:
        return Minute()

    def second(self) -> Second:
        return Second()

    def millisecond(self) -> Millisecond:
        return Millisecond()

    def microsecond(self) -> Microsecond:
        return Microsecond()

    def nanosecond(self) -> Nanosecond:
        return Nanosecond()

    def ordinal_day(self) -> OrdinalDay:
        return OrdinalDay()

    def weekday(self) -> WeekDay:
        return WeekDay()

    def total_minutes(self) -> TotalMinutes:
        return TotalMinutes()

    def total_seconds(self) -> TotalSeconds:
        return TotalSeconds()

    def total_milliseconds(self) -> TotalMilliseconds:
        return TotalMilliseconds()

    def total_microseconds(self) -> TotalMicroseconds:
        return TotalMicroseconds()

    def total_nanoseconds(self) -> TotalNanoseconds:
        return TotalNanoseconds()

    def to_string(self, format: str) -> ToString:
        return ToString(format=format)

    def replace_time_zone(self, time_zone: str | None) -> ReplaceTimeZone:
        return ReplaceTimeZone(time_zone=time_zone)

    def convert_time_zone(self, time_zone: str) -> ConvertTimeZone:
        return ConvertTimeZone(time_zone=time_zone)

    def timestamp(self, time_unit: TimeUnit = "us") -> Timestamp:
        return Timestamp.from_time_unit(time_unit)

    def truncate(self, every: str) -> Truncate:
        return Truncate.from_string(every)


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
        return self._with_unary(self._ir.timestamp(time_unit=time_unit))

    def truncate(self, every: str) -> Expr:
        return self._with_unary(self._ir.truncate(every=every))
