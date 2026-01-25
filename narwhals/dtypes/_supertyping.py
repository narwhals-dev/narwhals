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
from typing import TYPE_CHECKING, Any, Final, Generic, cast

from narwhals._constants import MS_PER_SECOND, NS_PER_SECOND, US_PER_SECOND
from narwhals._typing_compat import TypeVar
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

    from typing_extensions import TypeAlias, TypeIs

    from narwhals.dtypes import IntegerType
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

Incomplete: TypeAlias = Any
FrozenDTypes: TypeAlias = frozenset[type[DType]]
DTypeGroup: TypeAlias = frozenset[type[DType]]
Nested: TypeAlias = "Array | List | Struct"
Parametric: TypeAlias = (
    "Datetime | DatetimeV1 | Decimal |Duration | DurationV1 | Enum | Nested"
)
SameTemporalT = TypeVar("SameTemporalT", Datetime, DatetimeV1, Duration, DurationV1)
"""Temporal data types, with a `time_unit` attribute."""

SameDatetimeT = TypeVar("SameDatetimeT", Datetime, DatetimeV1)
SameT = TypeVar(
    "SameT",
    Array,
    List,
    Struct,
    Datetime,
    DatetimeV1,
    Decimal,
    Duration,
    DurationV1,
    Enum,
)
DTypeT1 = TypeVar("DTypeT1", bound=DType)
DTypeT2 = TypeVar("DTypeT2", bound=DType, default=DTypeT1)
DTypeT1_co = TypeVar("DTypeT1_co", bound=DType, covariant=True)
DTypeT2_co = TypeVar("DTypeT2_co", bound=DType, covariant=True, default=DTypeT1_co)


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


_TIME_UNIT_PER_SECOND: Mapping[TimeUnit, int] = {
    "s": 1,
    "ms": MS_PER_SECOND,
    "us": US_PER_SECOND,
    "ns": NS_PER_SECOND,
}


def _key_fn_time_unit(obj: Datetime | Duration, /) -> int:
    return _TIME_UNIT_PER_SECOND[obj.time_unit]


@lru_cache(maxsize=_CACHE_SIZE * 2)
def downcast_time_unit(
    left: SameTemporalT, right: SameTemporalT, /
) -> SameTemporalT | None:
    """Return the operand with the lowest precision time unit."""
    return min(left, right, key=_key_fn_time_unit)


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


def _struct_supertype(left: Struct, right: Struct, /) -> Struct | None:
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


def _array_supertype(left: Array, right: Array, /) -> Array | None:
    if (left.shape == right.shape) and (
        inner := get_supertype(left.inner(), right.inner())
    ):
        return Array(inner, left.size)
    return None


def _list_supertype(left: List, right: List, /) -> List | None:
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


def _decimal_supertype(left: Decimal, right: Decimal, /) -> Decimal:
    # https://github.com/pola-rs/polars/blob/529f7ec642912a2f15656897d06f1532c2f5d4c4/crates/polars-core/src/utils/supertype.rs#L508-L511
    precision = max(left.precision, right.precision)
    scale = max(left.scale, right.scale)
    return Decimal(precision=precision, scale=scale)


_SAME_DISPATCH: Final[Mapping[type[Parametric], Callable[..., Incomplete | None]]] = {
    Array: _array_supertype,
    List: _list_supertype,
    Struct: _struct_supertype,
    Datetime: _datetime_supertype,
    DatetimeV1: _datetime_supertype,
    Duration: downcast_time_unit,
    DurationV1: downcast_time_unit,
    Enum: _enum_supertype,
    Decimal: _decimal_supertype,
}
"""Specialized supertyping rules for `(T, T)`.

*When operands share the same class*, all other data types can use `DType.__eq__` (see [#3393]).

[#3393]: https://github.com/narwhals-dev/narwhals/pull/3393
"""


def is_single_base_type(
    st: _SupertypeCase[DTypeT1, DType],
) -> TypeIs[_SupertypeCase[DTypeT1]]:
    return len(st.base_types) == 1


def is_parametric_case(
    st: _SupertypeCase[SameT | DType],
) -> TypeIs[_SupertypeCase[SameT]]:
    return st.base_left in _SAME_DISPATCH


def _get_same_supertype_fn(base: type[SameT]) -> Callable[[SameT, SameT], SameT | None]:
    return cast("Callable[[SameT, SameT], SameT | None]", _SAME_DISPATCH[base])


def _same_supertype(st: _SupertypeCase[SameT | DType]) -> SameT | DType | None:
    if is_parametric_case(st):
        return _get_same_supertype_fn(st.base_left)(st.left, st.right)
    return st.left if dtype_eq(st.left, st.right) else None


DEC128_MAX_PREC = 38
# Precomputing powers of 10 up to 10^38
POW10_LIST = tuple(10**i for i in range(DEC128_MAX_PREC + 1))
INT_MAX_MAP: Mapping[IntegerType, int] = {
    UInt8(): (2**8) - 1,
    UInt16(): (2**16) - 1,
    UInt32(): (2**32) - 1,
    UInt64(): (2**64) - 1,
    Int8(): (2**7) - 1,
    Int16(): (2**15) - 1,
    Int32(): (2**31) - 1,
    Int64(): (2**63) - 1,
}


def dec128_fits(x: int, precision: int) -> bool:
    """Returns whether the given integer fits in the given precision."""
    return True if precision >= DEC128_MAX_PREC else abs(x) < POW10_LIST[precision]


def i128_to_dec128(x: int, precision: int, scale: int) -> int | None:
    """Scales an integer and checks if it fits the target precision."""
    # In Python, x * 10**s won't overflow like i128
    res = x * POW10_LIST[scale]  # Safe since scale <= precision <= 38
    return res if dec128_fits(res, precision) else None


def _decimal_integer_supertyping(decimal: Decimal, integer: IntegerType) -> DType | None:
    precision, scale = decimal.precision, decimal.scale

    def fits(v: int) -> bool:
        return i128_to_dec128(v, precision, scale) is not None

    if integer in {UInt128(), Int128()}:
        fits_orig_prec_scale = False
    elif value := INT_MAX_MAP.get(integer, None):
        fits_orig_prec_scale = fits(value)
    else:  # pragma: no cover
        msg = "Unreachable integer type"
        raise ValueError(msg)

    precision = precision if fits_orig_prec_scale else DEC128_MAX_PREC
    return Decimal(precision, scale)


@lru_cache(maxsize=_CACHE_SIZE)
def _numeric_supertype(st: _SupertypeCase[DType]) -> DType | None:
    """Get the supertype of two numeric data types that do not share the same class.

    `_{primitive_numeric,integer}_supertyping` define most valid numeric supertypes.

    We generate these on first use, with all subsequent calls returning the same mapping.
    """
    base_types = st.base_types
    if NUMERIC.issuperset(base_types):
        if INTEGER.issuperset(base_types):
            return _integer_supertyping()[base_types]()
        if tp := _FLOAT_PROMOTE.get(base_types):
            return tp()
        if Decimal in base_types:
            # Logic adapted from rust implementation
            # https://github.com/pola-rs/polars/blob/529f7ec642912a2f15656897d06f1532c2f5d4c4/crates/polars-core/src/utils/supertype.rs#L517-L530
            decimal, integer = (
                (st.left, st.right)
                if isinstance(st.left, Decimal)
                else (st.right, st.left)
            )
            return _decimal_integer_supertyping(decimal=decimal, integer=integer)  # type: ignore[arg-type]

        return _primitive_numeric_supertyping()[base_types]()
    if Boolean in base_types:
        return _first_excluding(base_types, Boolean)()
    return None


def _mixed_supertype(st: _SupertypeCase[DType, DType]) -> DType | None:
    """Get the supertype of two data types that do not share the same class."""
    base_types = st.base_types
    if Date in base_types and _has_intersection(base_types, DATETIME):
        return st.left if isinstance(st.left, Datetime) else st.right
    if NUMERIC.isdisjoint(base_types):
        return tp() if (tp := _STRING_LIKE_CONVERT.get(base_types)) else None
    return None if has_nested(base_types) else _numeric_supertype(st)


class _SupertypeCase(Generic[DTypeT1_co, DTypeT2_co]):
    """WIP."""

    __slots__ = ("base_types", "left", "right")

    left: DTypeT1_co
    right: DTypeT2_co
    base_types: frozenset[type[DTypeT1_co | DTypeT2_co]]

    def __init__(self, left: DTypeT1_co, right: DTypeT2_co) -> None:
        self.left = left
        self.right = right
        self.base_types = frozenset((self.base_left, self.base_right))

    @property
    def base_left(self) -> type[DTypeT1_co]:
        return self.left.base_type()

    @property
    def base_right(self) -> type[DTypeT2_co]:
        return self.right.base_type()


# NOTE @dangotbanned: Tried **many** variants of this typing
# (to self) DO NOT TOUCH IT AGAIN
def get_supertype(
    left: DTypeT1, right: DTypeT2 | DType
) -> DTypeT1 | DTypeT2 | DType | None:
    """Given two data types, determine the data type that both types can reasonably safely be cast to.

    Arguments:
        left: First data type.
        right: Second data type.

    Returns:
        The common supertype that both types can be safely cast to, or None if no such type exists.
    """
    st_case = _SupertypeCase(left, right)
    if Unknown in st_case.base_types:
        return Unknown()
    if is_single_base_type(st_case):
        return _same_supertype(st_case)
    return _mixed_supertype(st_case)
