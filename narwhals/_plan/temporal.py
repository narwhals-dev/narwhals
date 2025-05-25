from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from narwhals._plan.common import ExprNamespace, Function, IRNamespace
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from typing_extensions import TypeAlias, TypeIs

    from narwhals._duration import IntervalUnit
    from narwhals._plan.dummy import DummyExpr
    from narwhals.typing import TimeUnit

PolarsTimeUnit: TypeAlias = Literal["ns", "us", "ms"]


def _is_polars_time_unit(obj: Any) -> TypeIs[PolarsTimeUnit]:
    return obj in {"ns", "us", "ms"}


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

    time_unit: PolarsTimeUnit

    @staticmethod
    def from_time_unit(time_unit: TimeUnit, /) -> Timestamp:
        if not _is_polars_time_unit(time_unit):
            from typing import get_args

            msg = (
                "invalid `time_unit`"
                f"\n\nExpected one of {get_args(PolarsTimeUnit)}, got {time_unit!r}."
            )
            raise ValueError(msg)
        return Timestamp(time_unit=time_unit)


class Truncate(TemporalFunction):
    __slots__ = ("multiple", "unit")

    multiple: int
    unit: IntervalUnit

    @staticmethod
    def from_string(every: str, /) -> Truncate:
        from narwhals._duration import parse_interval_string

        multiple, unit = parse_interval_string(every)
        return Truncate(multiple=multiple, unit=unit)


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

    def date(self) -> DummyExpr:
        return self._to_narwhals(self._ir.date().to_function_expr(self._expr._ir))

    def year(self) -> DummyExpr:
        return self._to_narwhals(self._ir.year().to_function_expr(self._expr._ir))

    def month(self) -> DummyExpr:
        return self._to_narwhals(self._ir.month().to_function_expr(self._expr._ir))

    def day(self) -> DummyExpr:
        return self._to_narwhals(self._ir.day().to_function_expr(self._expr._ir))

    def hour(self) -> DummyExpr:
        return self._to_narwhals(self._ir.hour().to_function_expr(self._expr._ir))

    def minute(self) -> DummyExpr:
        return self._to_narwhals(self._ir.minute().to_function_expr(self._expr._ir))

    def second(self) -> DummyExpr:
        return self._to_narwhals(self._ir.second().to_function_expr(self._expr._ir))

    def millisecond(self) -> DummyExpr:
        return self._to_narwhals(self._ir.millisecond().to_function_expr(self._expr._ir))

    def microsecond(self) -> DummyExpr:
        return self._to_narwhals(self._ir.microsecond().to_function_expr(self._expr._ir))

    def nanosecond(self) -> DummyExpr:
        return self._to_narwhals(self._ir.nanosecond().to_function_expr(self._expr._ir))

    def ordinal_day(self) -> DummyExpr:
        return self._to_narwhals(self._ir.ordinal_day().to_function_expr(self._expr._ir))

    def weekday(self) -> DummyExpr:
        return self._to_narwhals(self._ir.weekday().to_function_expr(self._expr._ir))

    def total_minutes(self) -> DummyExpr:
        return self._to_narwhals(
            self._ir.total_minutes().to_function_expr(self._expr._ir)
        )

    def total_seconds(self) -> DummyExpr:
        return self._to_narwhals(
            self._ir.total_seconds().to_function_expr(self._expr._ir)
        )

    def total_milliseconds(self) -> DummyExpr:
        return self._to_narwhals(
            self._ir.total_milliseconds().to_function_expr(self._expr._ir)
        )

    def total_microseconds(self) -> DummyExpr:
        return self._to_narwhals(
            self._ir.total_microseconds().to_function_expr(self._expr._ir)
        )

    def total_nanoseconds(self) -> DummyExpr:
        return self._to_narwhals(
            self._ir.total_nanoseconds().to_function_expr(self._expr._ir)
        )

    def to_string(self, format: str) -> DummyExpr:
        return self._to_narwhals(
            self._ir.to_string(format=format).to_function_expr(self._expr._ir)
        )

    def replace_time_zone(self, time_zone: str | None) -> DummyExpr:
        return self._to_narwhals(
            self._ir.replace_time_zone(time_zone=time_zone).to_function_expr(
                self._expr._ir
            )
        )

    def convert_time_zone(self, time_zone: str) -> DummyExpr:
        return self._to_narwhals(
            self._ir.convert_time_zone(time_zone=time_zone).to_function_expr(
                self._expr._ir
            )
        )

    def timestamp(self, time_unit: TimeUnit = "us") -> DummyExpr:
        return self._to_narwhals(
            self._ir.timestamp(time_unit=time_unit).to_function_expr(self._expr._ir)
        )

    def truncate(self, every: str) -> DummyExpr:
        return self._to_narwhals(
            self._ir.truncate(every=every).to_function_expr(self._expr._ir)
        )
