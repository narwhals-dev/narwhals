"""Rules for safe type promotion.

Follows a subset of `polars`' [`get_supertype_with_options`].

See [Data type promotion rules] for an in-depth explanation.

[`get_supertype_with_options`]: https://github.com/pola-rs/polars/blob/529f7ec642912a2f15656897d06f1532c2f5d4c4/crates/polars-core/src/utils/supertype.rs#L142-L543
[Data type promotion rules]: https://narwhals-dev.github.io/narwhals/concepts/promotion-rules/
"""

from __future__ import annotations

from collections import deque
from itertools import chain, product
from operator import attrgetter
from typing import TYPE_CHECKING, Any

from narwhals._constants import MS_PER_SECOND, NS_PER_SECOND, US_PER_SECOND
from narwhals._dispatch import just_dispatch
from narwhals._typing_compat import TypeVar
from narwhals.dtypes._classes import (
    Array,
    Binary,
    Boolean,
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
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Unknown,
    UnsignedIntegerType,
)
from narwhals.dtypes._classes_v1 import Datetime as DatetimeV1, Duration as DurationV1

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from typing_extensions import TypeAlias, TypeIs

    from narwhals.dtypes._classes import _Bits
    from narwhals.typing import TimeUnit

    _Fn = TypeVar("_Fn", bound=Callable[..., Any])

    # NOTE: Hack to make `functools.cache` *not* negatively impact typing
    def cache(fn: _Fn, /) -> _Fn:
        return fn

    # NOTE: Double hack + pretends `maxsize` is keyword-only and has no default
    def lru_cache(*, maxsize: int) -> Callable[[_Fn], _Fn]:  # noqa: ARG001
        return cache
else:
    from functools import cache, lru_cache

FrozenDTypes: TypeAlias = frozenset[type[DType]]
DTypeGroup: TypeAlias = frozenset[type[DType]]
SameTemporalT = TypeVar("SameTemporalT", Datetime, DatetimeV1, Duration, DurationV1)
"""Temporal data types, with a `time_unit` attribute."""
SameDatetimeT = TypeVar("SameDatetimeT", Datetime, DatetimeV1)


def frozen_dtypes(*dtypes: type[DType]) -> FrozenDTypes:
    """Alternative `frozenset` constructor.

    Gets `mypy` to stop inferring a more precise type (that later becomes incompatible).
    """
    return frozenset(dtypes)


_CACHE_SIZE = 32
"""Arbitrary size (currently).

- 27 concrete `DType` classes
- 3 (V1) subclasses
- Pairwise comparisons, but order (of classes) is not important
"""


SIGNED_INTEGER: DTypeGroup = frozenset((Int8, Int16, Int32, Int64, Int128))
UNSIGNED_INTEGER: DTypeGroup = frozenset((UInt8, UInt16, UInt32, UInt64, UInt128))
INTEGER: DTypeGroup = SIGNED_INTEGER.union(UNSIGNED_INTEGER)
FLOAT: DTypeGroup = frozenset((Float32, Float64))
NUMERIC: DTypeGroup = FLOAT.union(INTEGER).union((Decimal,))
NESTED: DTypeGroup = frozenset((Struct, List, Array))
DATETIME: DTypeGroup = frozen_dtypes(Datetime, DatetimeV1)

_FLOAT_PROMOTE: Mapping[FrozenDTypes, type[Float64]] = {
    frozen_dtypes(Float32, Float64): Float64,
    frozen_dtypes(Decimal, Float64): Float64,
    frozen_dtypes(Decimal, Float32): Float64,
}


_TIME_UNIT_PER_SECOND: Mapping[TimeUnit, int] = {
    "s": 1,
    "ms": MS_PER_SECOND,
    "us": US_PER_SECOND,
    "ns": NS_PER_SECOND,
}


def _key_fn_time_unit(obj: Datetime | Duration, /) -> int:
    return _TIME_UNIT_PER_SECOND[obj.time_unit]


@lru_cache(maxsize=_CACHE_SIZE // 2)
def dtype_eq(left: DType, right: DType, /) -> bool:
    return left == right


@cache
def _integer_supertyping() -> Mapping[FrozenDTypes, type[Int | Float64]]:
    """Generate the supertype conversion table for all integer data type pairs."""
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
    """Generate the supertype conversion table for all (integer, float) data type pairs."""
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


@lru_cache(maxsize=_CACHE_SIZE)
def has_nested(base_types: FrozenDTypes, /) -> bool:
    return _has_intersection(base_types, NESTED)


def has_inner(dtype: Any) -> TypeIs[Array | List]:
    return isinstance(dtype, (Array, List))


@just_dispatch(upper_bound=DType)
def same_supertype(left: DType, right: DType, /) -> DType | None:
    return left if dtype_eq(left, right) else None


@same_supertype.register(Duration, DurationV1)
@lru_cache(maxsize=_CACHE_SIZE * 2)
def downcast_time_unit(left: SameTemporalT, right: SameTemporalT, /) -> SameTemporalT:
    """Return the operand with the lowest precision time unit."""
    return min(left, right, key=_key_fn_time_unit)


def _struct_fields_union(
    left: Collection[Field], right: Collection[Field], /
) -> Struct | None:
    """Adapted from [`union_struct_fields`](https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L559-L586)."""
    longest, shortest = (left, right) if len(left) >= len(right) else (right, left)
    longest_map = {f.name: f.dtype() for f in longest}
    for f in shortest:
        name, dtype = f.name, f.dtype()
        dtype_longest = longest_map.setdefault(name, dtype)
        if not dtype_eq(dtype, dtype_longest):
            if supertype := get_supertype(dtype, dtype_longest):
                longest_map[name] = supertype
            else:
                return None
    return Struct(longest_map)


@same_supertype.register(Struct)
def struct_supertype(left: Struct, right: Struct, /) -> Struct | None:
    """Get the supertype of two struct data types.

    Adapted from [`super_type_structs`](https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L588-L603)
    """
    left_fields, right_fields = left.fields, right.fields
    if len(left_fields) != len(right_fields):
        return _struct_fields_union(left_fields, right_fields)
    new_fields = deque["Field"]()
    for left_f, right_f in zip(left_fields, right_fields):
        if left_f.name != right_f.name:
            return _struct_fields_union(left_fields, right_fields)
        if supertype := get_supertype(left_f.dtype(), right_f.dtype()):
            new_fields.append(Field(left_f.name, supertype))
        else:
            return None
    return Struct(new_fields)


@same_supertype.register(Array)
def array_supertype(left: Array, right: Array, /) -> Array | None:
    if (left.shape == right.shape) and (
        inner := get_supertype(left.inner(), right.inner())
    ):
        return Array(inner, left.size)
    return None


@same_supertype.register(List)
def list_supertype(left: List, right: List, /) -> List | None:
    if inner := get_supertype(left.inner(), right.inner()):
        return List(inner)
    return None


@same_supertype.register(Datetime, DatetimeV1)
def datetime_supertype(
    left: SameDatetimeT, right: SameDatetimeT, /
) -> SameDatetimeT | None:
    if left.time_zone != right.time_zone:
        return None
    return downcast_time_unit(left, right)


@same_supertype.register(Enum)
def enum_supertype(left: Enum, right: Enum, /) -> Enum | None:
    return left if left.categories == right.categories else None


@same_supertype.register(Decimal)
def decimal_supertype(left: Decimal, right: Decimal, /) -> Decimal:
    # https://github.com/pola-rs/polars/blob/529f7ec642912a2f15656897d06f1532c2f5d4c4/crates/polars-core/src/utils/supertype.rs#L508-L511
    precision = max(left.precision, right.precision)
    scale = max(left.scale, right.scale)
    return Decimal(precision=precision, scale=scale)


DEC128_MAX_PREC = 38
# Precomputing powers of 10 up to 10^38
POW10_LIST = tuple(10**i for i in range(DEC128_MAX_PREC + 1))
INT_MAX_MAP: Mapping[Int, int] = {
    UInt8(): (2**8) - 1,
    UInt16(): (2**16) - 1,
    UInt32(): (2**32) - 1,
    UInt64(): (2**64) - 1,
    Int8(): (2**7) - 1,
    Int16(): (2**15) - 1,
    Int32(): (2**31) - 1,
    Int64(): (2**63) - 1,
}


def _integer_fits_in_decimal(value: int, precision: int, scale: int) -> bool:
    """Scales an integer and checks if it fits the target precision."""
    # !NOTE: Indexing is safe since `scale <= precision <= 38`
    return (precision == DEC128_MAX_PREC) or (
        value * POW10_LIST[scale] < POW10_LIST[precision]
    )


def _decimal_integer_supertyping(decimal: Decimal, integer: Int) -> DType | None:
    precision, scale = decimal.precision, decimal.scale

    if integer in {UInt128(), Int128()}:
        fits_orig_prec_scale = False
    elif value := INT_MAX_MAP.get(integer, None):
        fits_orig_prec_scale = _integer_fits_in_decimal(value, precision, scale)
    else:  # pragma: no cover
        msg = "Unreachable integer type"
        raise ValueError(msg)

    precision = precision if fits_orig_prec_scale else DEC128_MAX_PREC
    return Decimal(precision, scale)


def _numeric_supertype(
    left: DType, right: DType, base_types: FrozenDTypes
) -> DType | None:
    """Get the supertype of two numeric data types that do not share the same class.

    `_{primitive_numeric,integer}_supertyping` define most valid numeric supertypes.

    We generate these on first use, with all subsequent calls returning the same mapping.
    """
    if NUMERIC.issuperset(base_types):
        if INTEGER.issuperset(base_types):
            return _integer_supertyping()[base_types]()
        if tp := _FLOAT_PROMOTE.get(base_types):
            return tp()
        if Decimal in base_types:
            # Logic adapted from rust implementation
            # https://github.com/pola-rs/polars/blob/529f7ec642912a2f15656897d06f1532c2f5d4c4/crates/polars-core/src/utils/supertype.rs#L517-L530
            decimal, integer = (
                (left, right) if isinstance(left, Decimal) else (right, left)
            )
            return _decimal_integer_supertyping(decimal=decimal, integer=integer)  # type: ignore[arg-type]

        return _primitive_numeric_supertyping()[base_types]()
    if Boolean in base_types:
        return _first_excluding(base_types, Boolean)()
    return None


def _mixed_nested_supertype(left: DType, right: DType, /) -> DType | None:
    if (
        has_inner(left)
        and has_inner(right)
        and (inner := get_supertype(left.inner(), right.inner()))
    ):
        return List(inner)
    return None


def _mixed_supertype(
    left: DType, right: DType, base_types: FrozenDTypes, /
) -> DType | None:
    """Get the supertype of two data types that do not share the same class."""
    if Date in base_types and _has_intersection(base_types, DATETIME):
        return left if isinstance(left, Datetime) else right
    if String in base_types:
        return (Binary if Binary in base_types else String)()
    if has_nested(base_types):
        return _mixed_nested_supertype(left, right)
    return (
        _numeric_supertype(left, right, base_types)
        if _has_intersection(NUMERIC, base_types)
        else None
    )


def get_supertype(left: DType, right: DType) -> DType | None:
    """Given two data types, determine the data type that both types can reasonably safely be cast to.

    Arguments:
        left: First data type.
        right: Second data type.

    Returns:
        The common supertype that both types can be safely cast to, or None if no such type exists.
    """
    base_types = frozen_dtypes(left.base_type(), right.base_type())
    if Unknown in base_types:
        return Unknown()
    if len(base_types) == 1:
        return same_supertype(left, right)
    return _mixed_supertype(left, right, base_types)
