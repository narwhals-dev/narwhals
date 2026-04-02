from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeVar, get_args

import narwhals._plan.dtypes_mapper as dtm
from narwhals._duration import Interval
from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._flags import FunctionFlags
from narwhals._plan._function import Function
from narwhals._plan.expressions.namespace import IRNamespace
from narwhals.exceptions import ComputeError

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias

    from narwhals._duration import IntervalUnit
    from narwhals._plan.expressions import FunctionExpr as FExpr
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType
    from narwhals.typing import TimeUnit

PolarsTimeUnit: TypeAlias = Literal["ns", "us", "ms"]
_POLARS_TIME_UNIT = frozenset[PolarsTimeUnit](("ns", "us", "ms"))
Tz = TypeVar("Tz", str, "str | None")

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
ELEMENTWISE = FunctionFlags.ELEMENTWISE
same_dtype = ResolveDType.function.same_dtype


# fmt: off
class TemporalFunction(Function, dispatch=DispatcherOptions(accessor_name="dt"), flags=ELEMENTWISE): ...
class _TemporalInt8(TemporalFunction, dtype=dtm.I8): ...
class _TemporalInt32(TemporalFunction, dtype=dtm.I32): ...
class _TemporalInt64(TemporalFunction, dtype=dtm.I64): ...
class _TemporalTimeZone(TemporalFunction, Generic[Tz]):
    __slots__ = ("time_zone",)
    time_zone: Tz
    def resolve_dtype(self, node: FExpr[Self], schema: FrozenSchema, /) -> DType:  # pragma: no cover
        dtype = node.input[0].resolve_dtype(schema)
        if isinstance(dtype, dtm.dtypes.Datetime):
            return type(dtype)(dtype.time_unit, self.time_zone)
        msg = f"Expected Datetime, got {dtype}"
        raise ComputeError(msg)

class _TemporalInterval(TemporalFunction, dtype=same_dtype()):
    __slots__ = ("multiple", "unit")
    multiple: int
    unit: IntervalUnit
    @classmethod
    def from_string(cls, interval: str, /) -> Self:
        return cls.from_interval(Interval.parse(interval))
    @classmethod
    def from_interval(cls, interval: Interval, /) -> Self:
        return cls(multiple=interval.multiple, unit=interval.unit)

class Year(_TemporalInt32): ...
class Month(_TemporalInt8): ...
class Day(_TemporalInt8): ...
class Hour(_TemporalInt8): ...
class Minute(_TemporalInt8): ...
class Second(_TemporalInt8): ...
class Millisecond(_TemporalInt32): ...
class Microsecond(_TemporalInt32): ...
class Nanosecond(_TemporalInt32): ...
class OrdinalDay(TemporalFunction, dtype=dtm.I16): ...
class WeekDay(_TemporalInt8): ...
class TotalMinutes(_TemporalInt64): ...
class TotalSeconds(_TemporalInt64): ...
class TotalMilliseconds(_TemporalInt64): ...
class TotalMicroseconds(_TemporalInt64): ...
class TotalNanoseconds(_TemporalInt64): ...
class ReplaceTimeZone(_TemporalTimeZone["str | None"]): ...
class ConvertTimeZone(_TemporalTimeZone[str]): ...
class Truncate(_TemporalInterval): ...
class OffsetBy(_TemporalInterval): ...
class Date(TemporalFunction, dtype=dtm.DATE):...
class ToString(TemporalFunction, dtype=dtm.STR):
    __slots__ = ("format",)
    format: str
class Timestamp(_TemporalInt64):
    __slots__ = ("time_unit",)
    time_unit: PolarsTimeUnit
    @staticmethod
    def from_time_unit(time_unit: TimeUnit = "us", /) -> Timestamp:
        if time_unit in _POLARS_TIME_UNIT:
            # Needs `mypy>=1.20`
            # https://mypy.readthedocs.io/en/stable/changelog.html#better-type-narrowing
            return Timestamp(time_unit=time_unit) # type: ignore[arg-type]
        msg = f"Only the following time units are supported: {get_args(PolarsTimeUnit)}.\nGot: {time_unit!r}."
        raise TypeError(msg)
    def __repr__(self) -> str:
        return f"{super().__repr__()}[{self.time_unit!r}]"
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
