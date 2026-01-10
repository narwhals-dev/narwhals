from __future__ import annotations

from collections import deque
from itertools import chain, product
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Final, Protocol, TypeVar, cast

from narwhals.dtypes._classes import (
    Array,
    Binary,
    Boolean,
    Categorical,
    Date,
    Datetime,
    Decimal,
    DType,
    Duration,
    Enum,
    Field,
    Float32,
    Float64,
    FloatType as Float,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    IntegerType as Int,
    List,
    SignedIntegerType,
    String,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Unknown,
    UnsignedIntegerType,
)
from narwhals.dtypes._classes_v1 import (
    Datetime as DatetimeV1,
    Duration as DurationV1,
    Enum as EnumV1,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from typing_extensions import TypeAlias

    from narwhals.dtypes._classes import _Bits
    from narwhals.typing import TimeUnit

    class HasTimeUnit(Protocol):
        time_unit: TimeUnit

    HasTimeUnitT = TypeVar("HasTimeUnitT", bound=HasTimeUnit)
    SameDatetimeT = TypeVar("SameDatetimeT", Datetime, DatetimeV1)

    _Fn = TypeVar("_Fn", bound=Callable[..., Any])

    # NOTE: Hack to make `functools.cache` *not* negatively impact typing
    def cache(fn: _Fn, /) -> _Fn:
        return fn

    # NOTE: Double hack + pretends `maxsize` is keyword-only and has no default
    def lru_cache(*, maxsize: int) -> Callable[[_Fn], _Fn]:  # noqa: ARG001
        return cache
else:
    from functools import cache, lru_cache


def frozen_dtypes(*dtypes: type[DType]) -> FrozenDTypes:
    """Alternative `frozenset` constructor.

    Gets `mypy` to stop inferring a more precise type (that later becomes incompatible).
    """
    return frozenset(dtypes)


_CACHE_SIZE_TP_MID = 32
"""Arbitrary size (currently).

- 27 concrete `DType` classes
- 3 (V1) subclasses
- Pairwise comparisons, but order (of classes) is not important
"""

# TODO @dangotbanned: If this stays in, it needs docs
OpaqueDispatchFn: TypeAlias = "Callable[[DType, DType], DType | None]"

FrozenDTypes: TypeAlias = frozenset[type[DType]]
DTypeGroup: TypeAlias = frozenset[type[DType]]


SIGNED_INTEGER: DTypeGroup = frozenset((Int8, Int16, Int32, Int64, Int128))
UNSIGNED_INTEGER: DTypeGroup = frozenset((UInt8, UInt16, UInt32, UInt64, UInt128))
INTEGER: DTypeGroup = SIGNED_INTEGER.union(UNSIGNED_INTEGER)
FLOAT: DTypeGroup = frozenset((Float32, Float64))
NUMERIC: DTypeGroup = FLOAT.union(INTEGER).union((Decimal,))
NESTED: DTypeGroup = frozenset((Struct, List, Array))
DATETIME: DTypeGroup = frozen_dtypes(Datetime, DatetimeV1)
TEMPORAL: DTypeGroup = DATETIME.union((Date, Time, Duration, DurationV1))
STRING: DTypeGroup = frozenset((String, Binary, Categorical, Enum, EnumV1))

_STRING_LIKE_CONVERT: Mapping[FrozenDTypes, type[String | Binary]] = {
    frozen_dtypes(String, Categorical): String,
    frozen_dtypes(String, Enum): String,
    frozen_dtypes(String, EnumV1): String,
    frozen_dtypes(String, Binary): Binary,
}
_FLOAT_PROMOTE: Mapping[FrozenDTypes, type[Float64]] = {
    frozen_dtypes(Float32, Float64): Float64,
    frozen_dtypes(Decimal, Float64): Float64,
    frozen_dtypes(Decimal, Float32): Float64,
}


# NOTE: polars has these ordered as `["ns", "μs", "ms"]`
# https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/datatypes/temporal/time_unit.rs#L9-L41
# That order would align with `_max_bits` using `max`, but represent a downcast vs an upcast
_TIME_UNIT_TO_INDEX: Mapping[TimeUnit, int] = {"s": 0, "ms": 1, "us": 2, "ns": 3}
"""Convert time unit to an index for comparison (larger = more precise)."""


def _key_fn_time_unit(obj: HasTimeUnit, /) -> int:
    return _TIME_UNIT_TO_INDEX[obj.time_unit]


@lru_cache(maxsize=_CACHE_SIZE_TP_MID * 2)
def downcast_time_unit(left: HasTimeUnitT, right: HasTimeUnitT) -> HasTimeUnitT:
    """Return the operand with the lowest precision time unit."""
    return min(left, right, key=_key_fn_time_unit)


@cache
def _integer_supertyping() -> Mapping[FrozenDTypes, type[Int | Float64]]:
    tps_int = SignedIntegerType.__subclasses__()
    tps_uint = UnsignedIntegerType.__subclasses__()
    get_bits: attrgetter[_Bits] = attrgetter("_bits")
    ints = (
        (frozen_dtypes(lhs, rhs), max(lhs, rhs, key=get_bits))
        for lhs, rhs in product(tps_int, repeat=2)
    )
    uints = (
        (frozen_dtypes(lhs, rhs), max(lhs, rhs, key=get_bits))
        for lhs, rhs in product(tps_uint, repeat=2)
    )
    # NOTE: `Float64` is here because `mypy` refuses to respect the last overload 😭
    # https://github.com/python/typeshed/blob/a564787bf23386e57338b750bf4733f3c978b701/stdlib/typing.pyi#L776-L781
    ubits_to_int: Mapping[_Bits, type[Int | Float64]] = {8: Int16, 16: Int32, 32: Int64}
    mixed = (
        (
            frozen_dtypes(int_, uint),
            int_ if int_._bits > uint._bits else ubits_to_int.get(uint._bits, Float64),
        )
        for int_, uint in product(tps_int, tps_uint)
    )
    return dict(chain(ints, uints, mixed))


@cache
def _primitive_numeric_supertyping() -> Mapping[FrozenDTypes, type[Float]]:
    F32, F64 = Float32, Float64  # noqa: N806
    small_int = (Int8, Int16, UInt8, UInt16)
    small_int_f32 = ((frozen_dtypes(tp, F32), F32) for tp in small_int)
    big_int_f32 = ((frozen_dtypes(tp, F32), F64) for tp in INTEGER.difference(small_int))
    int_f64 = ((frozen_dtypes(tp, F64), F64) for tp in INTEGER)
    return dict(chain(small_int_f32, big_int_f32, int_f64))


def _first_excluding(base_types: FrozenDTypes, *exclude: type[DType]) -> type[DType]:
    """Return an arbitrary element from base_types excluding the given types."""
    others = base_types.difference(exclude)
    return next(iter(others))


def _has_intersection(a: frozenset[Any], b: frozenset[Any], /) -> bool:
    """Return True if sets share at least one element."""
    return not a.isdisjoint(b)


@lru_cache(maxsize=_CACHE_SIZE_TP_MID)
def has_nested(base_types: FrozenDTypes, /) -> bool:
    return _has_intersection(base_types, NESTED)


def _struct_union_fields(
    left: Collection[Field], right: Collection[Field]
) -> Struct | None:
    # if equal length we also take the lhs
    # so that the lhs determines the order of the fields
    longest, shortest = (left, right) if len(left) >= len(right) else (right, left)
    longest_map = {f.name: f.dtype() for f in longest}
    for f in shortest:
        name, dtype = f.name, f.dtype()
        dtype_longest = longest_map.setdefault(name, dtype)
        if dtype != dtype_longest:
            if supertype := get_supertype(dtype, dtype_longest):
                longest_map[name] = supertype
            else:
                return None
    return Struct(longest_map)


def _struct_supertype(left: Struct, right: Struct) -> Struct | None:
    # https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L588-L603
    left_fields, right_fields = left.fields, right.fields
    if len(left_fields) != len(right_fields):
        return _struct_union_fields(left_fields, right_fields)
    new_fields = deque["Field"]()
    for a, b in zip(left_fields, right_fields):
        if a.name != b.name:
            return _struct_union_fields(left_fields, right_fields)
        if supertype := get_supertype(a.dtype(), b.dtype()):
            new_fields.append(Field(a.name, supertype))
        else:
            return None
    return Struct(new_fields)


def _array_supertype(left: Array, right: Array) -> Array | None:
    if (left.shape == right.shape) and (
        inner := get_supertype(left.inner(), right.inner())
    ):
        return Array(inner, left.size)
    return None


def _list_supertype(left: List, right: List) -> List | None:
    if inner := get_supertype(left.inner(), right.inner()):
        return List(inner)
    return None


def _datetime_supertype(
    left: SameDatetimeT, right: SameDatetimeT, /
) -> SameDatetimeT | None:
    if left.time_zone != right.time_zone:
        return None
    return downcast_time_unit(left, right)


def _enum_supertype(left: Enum, right: Enum, /) -> Enum | None:
    return left if left.categories == right.categories else None


# TODO @dangotbanned: Try to do this while staying in the type system
_NESTED_DISPATCH: Final[Mapping[type[DType], OpaqueDispatchFn]] = {
    Array: cast("OpaqueDispatchFn", _array_supertype),
    List: cast("OpaqueDispatchFn", _list_supertype),
    Struct: cast("OpaqueDispatchFn", _struct_supertype),
}
# TODO @dangotbanned: Try to do this while staying in the type system
# TODO @dangotbanned: Probably merge with `_NESTED_DISPATCH`, after tweaking `get_supertype` flow
_PARAMETRIC_DISPATCH: Final[Mapping[type[DType], OpaqueDispatchFn]] = {
    Datetime: cast("OpaqueDispatchFn", _datetime_supertype),
    DatetimeV1: cast("OpaqueDispatchFn", _datetime_supertype),
    Duration: cast("OpaqueDispatchFn", downcast_time_unit),
    DurationV1: cast("OpaqueDispatchFn", downcast_time_unit),
    Enum: cast("OpaqueDispatchFn", _enum_supertype),
}


@lru_cache(maxsize=_CACHE_SIZE_TP_MID)
def _numeric_supertype(base_types: FrozenDTypes) -> DType | None:
    # (Integer, Integer) -> Integer | Float64
    # (Float, Float) -> Float
    # (Integer, Float) -> Float
    #  * Small integers (Int8, Int16, UInt8, UInt16) + Float32 -> Float32
    #  * Larger integers (Int32+) + Float32 -> Float64
    #  * Any integer + Float64 -> Float64
    # (Decimal, {Integer, Decimal}) -> Decimal
    # (Decimal, Float) -> Float64
    # (Boolean, Numeric) -> Numeric
    if NUMERIC.issuperset(base_types):
        if INTEGER.issuperset(base_types):
            return _integer_supertyping()[base_types]()
        if tp := _FLOAT_PROMOTE.get(base_types):
            return tp()
        if Decimal in base_types:
            # TODO(FBruzzesi): Decimal will need to be addressed separately once
            # https://github.com/narwhals-dev/narwhals/pull/3377 is merged as we need to:
            #  * For (Decimal, Decimal) -> combine the scale and precision
            #  * For (Decimal, Integer) -> check if integer type fits in decimal
            #  * For (Decimal, Float) -> check if integer type fits in decimal
            return Decimal()
        return _primitive_numeric_supertyping()[base_types]()
    if Boolean in base_types:
        return _first_excluding(base_types, Boolean)()
    return None


def _mixed_supertype(left: DType, right: DType, base_types: FrozenDTypes) -> DType | None:
    # !NOTE: The following rules are known, but not planned to be implemented here
    # (Date, {UInt,Int,Float}{32,64}) -> {Int,Float}{32,64}
    # (Time, {Int,Float}{32,64}) -> {Int,Float}64
    # (Datetime, {UInt,Int,Float}{32,64}) -> {Int,Float}64
    # (Duration, {UInt,Int,Float}{32,64}) -> {Int,Float}64
    # See https://github.com/narwhals-dev/narwhals/issues/121
    if Date in base_types and _has_intersection(base_types, DATETIME):
        # Every *other* valid mix doesn't need instance attributes, like `Datetime` does
        return left if isinstance(left, Datetime) else right
    if NUMERIC.isdisjoint(base_types):
        return tp() if (tp := _STRING_LIKE_CONVERT.get(base_types)) else None
    return _numeric_supertype(base_types)


def get_supertype(left: DType, right: DType) -> DType | None:
    """Given two data types, determine the data type that both types can reasonably safely be cast to.

    Aims to follow the rules defined by [`polars_core::utils::supertype::get_supertype_with_options`].

    Arguments:
        left: First data type.
        right: Second data type.

    Returns:
        The common supertype that both types can be safely cast to, or None if no such type exists.

    [`polars_core::utils::supertype::get_supertype_with_options`]: https://github.com/pola-rs/polars/blob/529f7ec642912a2f15656897d06f1532c2f5d4c4/crates/polars-core/src/utils/supertype.rs#L142-L543
    """
    base_left, base_right = left.base_type(), right.base_type()
    base_types = frozen_dtypes(base_left, base_right)
    if Unknown in base_types:
        return Unknown()
    has_mixed = len(base_types) != 1
    if has_nested(base_types):
        # NOTE: There are some other branches for `(Struct, DType) -> Struct`
        # But we aren't planning to use those.
        # The order of these conditions means we swallow all `(Nested, Non-Nested)` here,
        # simplifying both `_NESTED_DISPATCH` and everything that hits `has_nested(base_types) -> False`
        return None if has_mixed else _NESTED_DISPATCH[base_left](left, right)
    if has_mixed:
        return _mixed_supertype(left, right, base_types)
    if parametric := _PARAMETRIC_DISPATCH.get(base_left):
        return parametric(left, right)
    # NOTE: See for why this *isn't* the first thing we do
    # https://github.com/narwhals-dev/narwhals/pull/3393
    return left if left == right else None
