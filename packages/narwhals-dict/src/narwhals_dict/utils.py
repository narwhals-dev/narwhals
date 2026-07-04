from __future__ import annotations

import enum
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from itertools import islice
from typing import TYPE_CHECKING, Any, cast

from narwhals.exceptions import InvalidOperationError, ShapeError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from narwhals._utils import Version
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType, TimeUnit
    from narwhals_dict.typing import NativeSeries

__all__ = [
    "binary_op",
    "cast_values",
    "infer_dtype",
    "is_native_column",
    "is_native_frame",
    "narwhals_to_native_dtype",
    "native_to_narwhals_dtype",
    "parse_datetime_format",
    "parse_time_format",
    "trunc_div",
]

INFER_SAMPLE_SIZE = 5
"""How many non-null values are inspected when inferring a column's dtype."""

EPOCH_DATE = date(1970, 1, 1)
EPOCH_NAIVE = datetime(1970, 1, 1)
EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)

MICROSECONDS_PER_UNIT: Mapping[str, int] = {"s": 1_000_000, "ms": 1_000, "us": 1}


def is_native_column(obj: Any) -> bool:
    """A native column is any non-string sequence, `None` marks a null."""
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


def is_native_frame(obj: Any) -> bool:
    return isinstance(obj, dict) and all(
        isinstance(name, str) and is_native_column(column) for name, column in obj.items()
    )


def trunc_div(numerator: int, denominator: int) -> int:
    """Integer division truncating toward zero (like Polars temporal downscaling)."""
    quotient, remainder = divmod(numerator, denominator)
    if quotient < 0 and remainder:
        return quotient + 1
    return quotient


# NOTE: `bool` before `int`, `datetime` before `date` (subclass relationships).
_PY_TYPE_TO_DTYPE_NAME: tuple[tuple[type[Any], str], ...] = (
    (bool, "Boolean"),
    (int, "Int64"),
    (float, "Float64"),
    (str, "String"),
    (datetime, "Datetime"),
    (date, "Date"),
    (time, "Time"),
    (timedelta, "Duration"),
    (bytes, "Binary"),
)


def native_to_narwhals_dtype(native_dtype: type[Any], version: Version) -> DType:
    """Map a native (Python) type to a Narwhals dtype.

    Parametric information that only exists on *values* (`Struct` fields, `List`
    inner type, `Datetime` time zone) is handled by `infer_dtype`; this mapping
    covers everything recoverable from the type object alone.
    """
    dtypes = version.dtypes
    if issubclass(native_dtype, enum.Enum):
        # NOTE: before everything else: e.g. `enum.IntEnum` is a subclass of `int`.
        return dtypes.Enum(native_dtype)
    for py_type, dtype_name in _PY_TYPE_TO_DTYPE_NAME:
        if issubclass(native_dtype, py_type):
            return getattr(dtypes, dtype_name)()
    return dtypes.Unknown()


def _dtype_of_value(value: Any, version: Version) -> DType:
    dtypes = version.dtypes
    if isinstance(value, datetime):
        time_zone = str(value.tzinfo) if value.tzinfo is not None else None
        return dtypes.Datetime("us", time_zone)
    if isinstance(value, Mapping):
        return dtypes.Struct(
            {
                name: dtypes.Unknown()
                if field is None
                else _dtype_of_value(field, version)
                for name, field in value.items()
            }
        )
    if is_native_column(value):
        return dtypes.List(infer_dtype(value, version))
    return native_to_narwhals_dtype(type(value), version)


def infer_dtype(values: Iterable[Any], version: Version) -> DType:
    """Infer the Narwhals dtype from the first `INFER_SAMPLE_SIZE` non-null values.

    Mixed `Int64`/`Float64` samples promote to `Float64`; any other mixture is
    reported as `Unknown`.
    """
    dtypes = version.dtypes
    non_null = (value for value in values if value is not None)
    first = next(non_null, None)
    if first is None:
        return dtypes.Unknown()
    if isinstance(first, Mapping) or is_native_column(first):
        # TODO(FBruzzesi): Mixed types in nested types are not parsed properly,
        # so infer nested dtypes from the first non-null element only.
        return _dtype_of_value(first, version)

    found: list[DType] = [_dtype_of_value(first, version)]
    for value in islice(non_null, INFER_SAMPLE_SIZE - 1):
        if (dtype := _dtype_of_value(value, version)) not in found:
            found.append(dtype)

    if len(found) == 1:
        return found[0]
    if all(dtype == dtypes.Int64() or dtype == dtypes.Float64() for dtype in found):
        return dtypes.Float64()
    return dtypes.Unknown()


def _as_instance(dtype: IntoDType) -> DType:
    return dtype() if isinstance(dtype, type) else dtype


@lru_cache(maxsize=4)
def _nw_to_py_types(version: Version) -> Mapping[type[DType], type[Any]]:
    dtypes = version.dtypes
    return {
        dtypes.String: str,
        dtypes.Categorical: str,
        dtypes.Boolean: bool,
        dtypes.Datetime: datetime,
        dtypes.Date: date,
        dtypes.Time: time,
        dtypes.Duration: timedelta,
        dtypes.Binary: bytes,
        dtypes.Struct: dict,
        dtypes.List: list,
    }


def narwhals_to_native_dtype(dtype: IntoDType, version: Version) -> type[Any]:
    """Map a Narwhals dtype to the native (Python) type used to represent it."""
    dtypes = version.dtypes
    dtype = _as_instance(dtype)
    if isinstance(dtype, dtypes.Enum):
        return enum.Enum("Enum", {category: category for category in dtype.categories})  # type: ignore[return-value]
    if py_type := _nw_to_py_types(version).get(dtype.base_type()):
        return py_type
    if dtype.is_integer():
        return int
    if dtype.is_float():
        return float
    msg = f"Converting to {dtype} dtype is not supported for the dict backend."
    raise NotImplementedError(msg)


def datetime_to_us(value: datetime, /) -> int:
    """Exact microseconds since epoch (no float round-tripping)."""
    delta = value - (EPOCH_UTC if value.tzinfo is not None else EPOCH_NAIVE)
    return timedelta_to_us(delta)


def timedelta_to_us(value: timedelta, /) -> int:
    return (value.days * 86_400 + value.seconds) * MICROSECONDS_PER_UNIT[
        "s"
    ] + value.microseconds


def _to_int(value: Any) -> int:
    if isinstance(value, (date, time, timedelta)):
        msg = "Casting temporal values to numeric dtypes is not supported for the dict backend."
        raise InvalidOperationError(msg)
    return int(value)


def _to_str(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, enum.Enum):
        return str(value.value)
    return str(value)


def _check_time_unit(time_unit: TimeUnit, dtype: DType) -> int:
    if time_unit not in MICROSECONDS_PER_UNIT:
        msg = (
            f"Casting to {dtype} is not supported for the dict backend: "
            "Python datetime objects are microsecond-precision."
        )
        raise NotImplementedError(msg)
    return MICROSECONDS_PER_UNIT[time_unit]


def _coerce_to_datetime(value: Any, us_per_unit: int, dtype: DType) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, int):
        return EPOCH_NAIVE + timedelta(microseconds=value * us_per_unit)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            msg = f"Conversion from `str` value {value!r} to `datetime` failed."
            raise InvalidOperationError(msg) from exc
    msg = f"Casting {type(value).__name__!r} values to {dtype} is not supported."
    raise InvalidOperationError(msg)


def _truncate_to_unit(value: datetime, time_unit: TimeUnit) -> datetime:
    if time_unit == "s":
        return value.replace(microsecond=0)
    if time_unit == "ms":
        factor = MICROSECONDS_PER_UNIT["ms"]
        return value.replace(microsecond=value.microsecond // factor * factor)
    return value


def _datetime_caster(
    time_unit: TimeUnit, time_zone: str | None, dtype: DType
) -> Callable[[Any], datetime]:
    from zoneinfo import ZoneInfo

    us_per_unit = _check_time_unit(time_unit, dtype)
    target_tz = ZoneInfo(time_zone) if time_zone is not None else None

    def caster(value: Any) -> datetime:
        result = _coerce_to_datetime(value, us_per_unit, dtype)
        # Physical values are UTC-based: attaching/removing a time zone converts.
        if target_tz is not None:
            result = (
                result.replace(tzinfo=timezone.utc).astimezone(target_tz)
                if result.tzinfo is None
                else result.astimezone(target_tz)
            )
        elif result.tzinfo is not None:
            result = result.astimezone(timezone.utc).replace(tzinfo=None)
        return _truncate_to_unit(result, time_unit)

    return caster


def _duration_caster(time_unit: TimeUnit, dtype: DType) -> Callable[[Any], timedelta]:
    us_per_unit = _check_time_unit(time_unit, dtype)

    def caster(value: Any) -> timedelta:
        if isinstance(value, timedelta):
            us = trunc_div(timedelta_to_us(value), us_per_unit) * us_per_unit
            return timedelta(microseconds=us)
        if isinstance(value, int):
            return timedelta(microseconds=value * us_per_unit)
        msg = f"Casting {type(value).__name__!r} values to {dtype} is not supported."
        raise InvalidOperationError(msg)

    return caster


def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, int):
        return EPOCH_DATE + timedelta(days=value)
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            msg = f"Conversion from `str` value {value!r} to `date` failed."
            raise InvalidOperationError(msg) from exc
    msg = f"Casting {type(value).__name__!r} values to Date is not supported."
    raise InvalidOperationError(msg)


def _to_time(value: Any) -> time:
    if isinstance(value, datetime):
        return value.time()
    if isinstance(value, time):
        return value
    if isinstance(value, int):  # nanoseconds since midnight
        microseconds, _ = divmod(value, 1_000)
        return (EPOCH_NAIVE + timedelta(microseconds=microseconds)).time()
    msg = f"Casting {type(value).__name__!r} values to Time is not supported."
    raise InvalidOperationError(msg)


def _to_binary(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode()
    msg = f"Casting {type(value).__name__!r} values to Binary is not supported."
    raise InvalidOperationError(msg)


def _enum_caster(categories: tuple[str, ...]) -> Callable[[Any], enum.Enum]:
    member_by_value = cast(
        "type[enum.Enum]",
        enum.Enum("Enum", {category: category for category in categories}),
    )

    def caster(value: Any) -> enum.Enum:
        raw = value.value if isinstance(value, enum.Enum) else value
        try:
            return member_by_value(raw)
        except ValueError as exc:
            msg = f"Value {raw!r} is not in the Enum categories: {list(categories)}."
            raise InvalidOperationError(msg) from exc

    return caster


def _struct_caster(
    fields: Mapping[str, IntoDType], version: Version
) -> Callable[[Any], dict[str, Any]]:
    casters = {name: _caster_for(dtype, version) for name, dtype in fields.items()}

    def caster(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            msg = f"Casting {type(value).__name__!r} values to Struct is not supported."
            raise InvalidOperationError(msg)
        return {
            name: None if (field := value.get(name)) is None else field_caster(field)
            for name, field_caster in casters.items()
        }

    return caster


@lru_cache(maxsize=4)
def _simple_casters(version: Version) -> Mapping[type[DType], Callable[[Any], Any]]:
    dtypes = version.dtypes
    # NOTE: Categorical is represented as plain strings; the dtype itself
    # cannot round-trip since inference is value-based.
    return {
        dtypes.String: _to_str,
        dtypes.Categorical: _to_str,
        dtypes.Boolean: bool,
        dtypes.Date: _to_date,
        dtypes.Time: _to_time,
        dtypes.Binary: _to_binary,
    }


def _caster_for(dtype: IntoDType, version: Version) -> Callable[[Any], Any]:
    dtypes = version.dtypes
    dtype = _as_instance(dtype)
    if caster := _simple_casters(version).get(dtype.base_type()):
        return caster
    if dtype.is_integer():
        return _to_int
    if dtype.is_float():
        return float
    if isinstance(dtype, dtypes.Enum):
        return _enum_caster(dtype.categories)
    if isinstance(dtype, dtypes.Datetime):
        return _datetime_caster(dtype.time_unit, dtype.time_zone, dtype)
    if isinstance(dtype, dtypes.Duration):
        return _duration_caster(dtype.time_unit, dtype)
    if isinstance(dtype, dtypes.Struct):
        return _struct_caster(dict(iter(dtype)), version)
    if isinstance(dtype, dtypes.List):
        inner = dtype.inner
        return lambda value: cast_values(value, inner, version)
    msg = f"`cast` to {dtype} is not supported for the dict backend."
    raise NotImplementedError(msg)


def cast_values(values: Iterable[Any], dtype: IntoDType, version: Version) -> list[Any]:
    """Cast values elementwise to `dtype`'s native representation, propagating nulls."""
    caster = _caster_for(dtype, version)
    return [None if value is None else caster(value) for value in values]


def binary_op(
    op: Callable[[Any, Any], Any], left: NativeSeries, right: Any, *, is_scalar: bool
) -> list[Any]:
    """Elementwise binary operation with null (`None`) propagation."""
    if is_scalar:
        if right is None:
            return [None] * len(left)
        return [None if lhs is None else op(lhs, right) for lhs in left]
    if len(left) != len(right):
        msg = f"Expected object of length {len(left)}, got: {len(right)}."
        raise ShapeError(msg)
    return [
        None if (lhs is None or rhs is None) else op(lhs, rhs)
        for lhs, rhs in zip(left, right, strict=True)
    ]


_FULL_DATETIME_RE = re.compile(
    r"""
    (?P<date>                            # date component
        \d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}  #   separated, e.g. 2020-01-02, 01/02/2020
        | \d{8}                          #   compact, e.g. 20200102
    )
    (?P<sep>\s|T)?                       # date/time separator: whitespace or 'T'
    (?P<time>                            # time component
        \d{2}:\d{2}(?::\d{2})?           #   separated, e.g. 12:34 or 12:34:56
        | \d{6}?                         #   compact, e.g. 123456
    )?
    (?P<tz>Z|[+-]\d{2}:?\d{2})?          # timezone: 'Z', '+02:00', '+0200'
    $
    """,
    re.VERBOSE,
)

_YEAR_RE = r"(?:[12][0-9])?[0-9]{2}"  # 2- or 4-digit year, e.g. 2020 or 20
_MONTH_RE = r"0[1-9]|1[0-2]"  # 01-12
_DAY_RE = r"0[1-9]|[12][0-9]|3[01]"  # 01-31

_DATE_FORMATS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            rf"""
            ^
            {_YEAR_RE}          # year
            (?:{_MONTH_RE})     # month
            (?:{_DAY_RE})       # day
            $
            """,
            re.VERBOSE,
        ),
        "%Y%m%d",
    ),
    (
        re.compile(
            rf"""
            ^
            (?:{_YEAR_RE})      # year
            (?P<sep1>[-/.])
            (?:{_MONTH_RE})     # month
            (?P<sep2>[-/.])
            (?:{_DAY_RE})       # day
            $
            """,
            re.VERBOSE,
        ),
        "%Y-%m-%d",
    ),
    (
        re.compile(
            rf"""
            ^
            (?:{_DAY_RE})       # day
            (?P<sep1>[-/.])
            (?:{_MONTH_RE})     # month
            (?P<sep2>[-/.])
            (?:{_YEAR_RE})      # year
            $
            """,
            re.VERBOSE,
        ),
        "%d-%m-%Y",
    ),
    (
        re.compile(
            rf"""
            ^
            (?:{_MONTH_RE})     # month
            (?P<sep1>[-/.])
            (?:{_DAY_RE})       # day
            (?P<sep2>[-/.])
            (?:{_YEAR_RE})      # year
            $
            """,
            re.VERBOSE,
        ),
        "%m-%d-%Y",
    ),
)
_TIME_FORMATS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\d{2}:\d{2}:\d{2}$"), "%H:%M:%S"),
    (re.compile(r"^\d{2}:\d{2}$"), "%H:%M"),
    (re.compile(r"^\d{6}$"), "%H%M%S"),
)


def _sample_strings(values: Iterable[Any]) -> list[str]:
    return list(islice((value for value in values if value is not None), 10))


def parse_datetime_format(values: Iterable[Any]) -> str:
    """Try to infer a datetime `strptime` format from string values."""
    sample = _sample_strings(values)
    matches = [_FULL_DATETIME_RE.match(value) for value in sample]
    if not sample or not all(matches):
        msg = "Unable to infer datetime format, provided format is not supported."
        raise NotImplementedError(msg)
    if len({match["sep"] for match in matches if match}) > 1:
        msg = "Found multiple separator values while inferring datetime format."
        raise ValueError(msg)
    if len({match["tz"] for match in matches if match}) > 1:
        msg = "Found multiple timezone values while inferring datetime format."
        raise ValueError(msg)
    first = matches[0]
    assert first is not None  # noqa: S101
    date_format = _parse_date_format([match["date"] for match in matches if match])
    time_format = parse_time_format(match["time"] for match in matches if match)
    separator = first["sep"] or ""
    tz_format = "%z" if first["tz"] else ""
    return f"{date_format}{separator}{time_format}{tz_format}"


def _parse_date_format(values: list[str]) -> str:
    for pattern, date_format in _DATE_FORMATS:
        matches = [pattern.match(value) for value in values]
        if not all(matches):
            continue
        if date_format == "%Y%m%d":
            return date_format
        separators = {(match["sep1"], match["sep2"]) for match in matches if match}
        if len(separators) == 1:
            sep1, sep2 = separators.pop()
            if sep1 == sep2:
                return date_format.replace("-", sep1)
    msg = "Unable to infer date format."
    raise ValueError(msg)


def parse_time_format(values: Iterable[Any]) -> str:
    """Try to infer a time `strptime` format from string values."""
    sample = _sample_strings(values)
    for pattern, time_format in _TIME_FORMATS:
        if sample and all(pattern.match(value) for value in sample):
            return time_format
    return ""
