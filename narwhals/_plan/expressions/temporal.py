from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import narwhals._plan.dtypes_mapper as dtm
from narwhals._duration import Interval
from narwhals._plan._function import Function
from narwhals._plan.expressions.namespace import ExprNamespace, IRNamespace
from narwhals._plan.options import FunctionOptions
from narwhals.exceptions import ComputeError

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias, TypeIs

    from narwhals._duration import IntervalUnit
    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions import FunctionExpr as FExpr
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType
    from narwhals.typing import TimeUnit

PolarsTimeUnit: TypeAlias = Literal["ns", "us", "ms"]


def _is_polars_time_unit(obj: Any) -> TypeIs[PolarsTimeUnit]:
    return obj in {"ns", "us", "ms"}


# fmt: off
class TemporalFunction(Function, accessor="dt", options=FunctionOptions.elementwise): ...
class Date(TemporalFunction): ... # Date
class Year(TemporalFunction): ... # Int32
class Month(TemporalFunction): ... # Int8
class Day(TemporalFunction): ... # Int8
class Hour(TemporalFunction): ... # Int8
class Minute(TemporalFunction): ... # Int8
class Second(TemporalFunction): ... # Int8
class Millisecond(TemporalFunction): ... # Int32
class Microsecond(TemporalFunction): ... # Int32
class Nanosecond(TemporalFunction): ... # Int32
class OrdinalDay(TemporalFunction): ... # Int16
class WeekDay(TemporalFunction): ... # Int8
class TotalMinutes(TemporalFunction): ... # Int64
class TotalSeconds(TemporalFunction): ... # Int64
class TotalMilliseconds(TemporalFunction): ... # Int64
class TotalMicroseconds(TemporalFunction): ... # Int64
class TotalNanoseconds(TemporalFunction): ... # Int64
# fmt: on
class ToString(TemporalFunction):  # String
    __slots__ = ("format",)
    format: str


class ReplaceTimeZone(TemporalFunction):  # map_datetime_dtype_timezone
    __slots__ = ("time_zone",)
    time_zone: str | None


class ConvertTimeZone(TemporalFunction):
    __slots__ = ("time_zone",)
    time_zone: str

    def _resolve_dtype(self, schema: FrozenSchema, node: FExpr[Function]) -> DType:
        dtype = node.input[0]._resolve_dtype(schema)
        if isinstance(dtype, dtm.dtypes.Datetime):
            return type(dtype)(dtype.time_unit, self.time_zone)
        msg = f"Expected Datetime, got {dtype}"
        raise ComputeError(msg)


class Timestamp(TemporalFunction):  # Int64
    __slots__ = ("time_unit",)
    time_unit: PolarsTimeUnit

    @staticmethod
    def from_time_unit(time_unit: TimeUnit = "us", /) -> Timestamp:
        if not _is_polars_time_unit(time_unit):
            msg = f"invalid `time_unit` \n\nExpected one of ['ns', 'us', 'ms'], got {time_unit!r}."
            raise TypeError(msg)
        return Timestamp(time_unit=time_unit)

    def __repr__(self) -> str:
        return f"{super().__repr__()}[{self.time_unit!r}]"


class _IntervalFunction(TemporalFunction):
    __slots__ = ("multiple", "unit")
    multiple: int
    unit: IntervalUnit

    @classmethod
    def from_string(cls, interval: str, /) -> Self:
        return cls.from_interval(Interval.parse(interval))

    @classmethod
    def from_interval(cls, interval: Interval, /) -> Self:
        return cls(multiple=interval.multiple, unit=interval.unit)

    def _resolve_dtype(self, schema: FrozenSchema, node: FExpr[Function]) -> DType:
        return node.input[0]._resolve_dtype(schema)


# fmt: off
class Truncate(_IntervalFunction): ...
class OffsetBy(_IntervalFunction): ...
# fmt: on


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
    offset_by: ClassVar = staticmethod(OffsetBy.from_string)
    truncate: ClassVar = staticmethod(Truncate.from_string)
    timestamp: ClassVar = staticmethod(Timestamp.from_time_unit)


class ExprDateTimeNamespace(ExprNamespace[IRDateTimeNamespace]):
    @property
    def _ir_namespace(self) -> type[IRDateTimeNamespace]:
        return IRDateTimeNamespace

    def date(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.date())

    def year(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.year())

    def month(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.month())

    def day(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.day())

    def hour(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.hour())

    def minute(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.minute())

    def second(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.second())

    def millisecond(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.millisecond())

    def microsecond(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.microsecond())

    def nanosecond(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.nanosecond())

    def ordinal_day(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.ordinal_day())

    def weekday(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.weekday())

    def total_minutes(self) -> Expr:
        return self._with_unary(self._ir.total_minutes())

    def total_seconds(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.total_seconds())

    def total_milliseconds(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.total_milliseconds())

    def total_microseconds(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.total_microseconds())

    def total_nanoseconds(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.total_nanoseconds())

    def to_string(self, format: str) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.to_string(format=format))

    def replace_time_zone(self, time_zone: str | None) -> Expr:
        return self._with_unary(self._ir.replace_time_zone(time_zone=time_zone))

    def convert_time_zone(self, time_zone: str) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.convert_time_zone(time_zone=time_zone))

    def timestamp(self, time_unit: TimeUnit = "us") -> Expr:
        return self._with_unary(self._ir.timestamp(time_unit))

    def truncate(self, every: str) -> Expr:
        return self._with_unary(self._ir.truncate(every))

    def offset_by(self, by: str) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.offset_by(by))
