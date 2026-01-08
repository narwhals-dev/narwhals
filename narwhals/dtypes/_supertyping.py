from __future__ import annotations

from collections import deque
from itertools import chain, product
from operator import attrgetter
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar

from narwhals.dtypes.classes import (  # NOTE: Should not include `DType`(s) that are versioned
    Binary,
    Decimal,
    Float64,
    Int16,
    Int32,
    Int64,
    NumericType as Numeric,
    SignedIntegerType,
    String,
    Unknown,
    UnsignedIntegerType,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from typing_extensions import TypeIs

    from narwhals.dtypes.classes import (
        Boolean,
        DType,
        Field,
        FloatType as Float,
        IntegerType as Int,
        Struct,
        _Bits,
    )
    from narwhals.typing import DTypes, TimeUnit

    class _HasBitsInst(Protocol):
        _bits: _Bits

    class _HasBitsType(Protocol):
        _bits: ClassVar[_Bits]

    HasBitsT = TypeVar("HasBitsT", bound="_HasBitsInst | _HasBitsType")

    _Fn = TypeVar("_Fn", bound=Callable[..., Any])

    # NOTE: Hack to make `functools.cache` *not* negatively impact typing
    def cache(fn: _Fn, /) -> _Fn:
        return fn
else:
    from functools import cache

# NOTE: polars has these ordered as `["ns", "μs", "ms"]`
# https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/datatypes/temporal/time_unit.rs#L9-L41
# That order would align with `_max_bits` using `max`, but represent a downcast vs an upcast
_TIME_UNIT_TO_INDEX: Mapping[TimeUnit, int] = {"s": 0, "ms": 1, "us": 2, "ns": 3}
"""Convert time unit to an index for comparison (larger = more precise)."""


# NOTE: `Float64` is here because `mypy` refuses to respect the last overload 😭
# https://github.com/python/typeshed/blob/a564787bf23386e57338b750bf4733f3c978b701/stdlib/typing.pyi#L776-L781
_U_BITS_TO_INT: Mapping[_Bits, type[Int | Float64]] = {8: Int16, 16: Int32, 32: Int64}

_get_bits: Callable[[_HasBitsInst | _HasBitsType], _Bits] = attrgetter("_bits")


# TODO @dangotbanned: Define the signatures inside `TYPE_CHECKING`,
# but implement using `operator.attrgetter` outside
# TODO @dangotbanned: (Alternative) just define these here as either:
#   - `issubclass(dtype.base_type(), NumericType)`
#   - `isinstance(dtype, NumericType)`
def is_numeric(dtype: DType) -> TypeIs[Numeric]:
    return dtype.is_numeric()


def is_float(dtype: DType) -> TypeIs[Float]:
    return dtype.is_float()


def is_integer(dtype: DType) -> TypeIs[Int]:
    return dtype.is_integer()


def is_boolean(dtype: DType) -> TypeIs[Boolean]:
    return dtype.is_boolean()


@cache
def _min_time_unit(a: TimeUnit, b: TimeUnit) -> TimeUnit:
    """Return the less precise time unit."""
    return min(a, b, key=_TIME_UNIT_TO_INDEX.__getitem__)


@cache
def _max_bits(left: HasBitsT, right: HasBitsT, /) -> HasBitsT:
    return max(left, right, key=_get_bits)


def _gen_same_signed(
    dtypes: Iterable[type[Int]],
) -> Iterator[tuple[frozenset[type[Int]], type[Int]]]:
    for left, right in product(dtypes, repeat=2):
        yield frozenset((left, right)), _max_bits(left, right)


@cache
def _integer_supertyping() -> Callable[[Int, Int], Int | Float64]:
    """Get supertype for two integer types.

    ### @FBruzzesi
    Following Polars rules:

    - Same signedness: return the larger type
    - Mixed signedness: promote to signed
      - If signed type is strictly larger than unsigned, it can hold both
    - Int64 + UInt64 -> Float64 (following Polars)
    - Fallback to Float64 if no integer type large enough

    ### @dangotbanned
    - Does a "big" job once
    - Then everything after is a single `dict` lookup
    - `frozenset` allows us to match flipped operands to the same key
    """
    tps_int = SignedIntegerType.__subclasses__()
    tps_uint = UnsignedIntegerType.__subclasses__()
    mixed = (
        (
            frozenset((int_, uint)),
            int_ if int_._bits > uint._bits else _U_BITS_TO_INT.get(uint._bits, Float64),
        )
        for int_, uint in product(tps_int, tps_uint)
    )
    lookup = dict(chain(_gen_same_signed(tps_int), _gen_same_signed(tps_uint), mixed))

    def promote(left: Int, right: Int, /) -> Int | Float64:
        return lookup[frozenset((left.base_type(), right.base_type()))]()

    return promote


def _struct_union_fields(
    left: list[Field], right: list[Field], /, *, dtypes: DTypes
) -> Struct | None:
    # https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L559-L586
    # if equal length we also take the lhs
    # so that the lhs determines the order of the fields
    if len(left) >= len(right):
        longest, shortest = left, right
    else:
        longest, shortest = right, left
    longest_map = {f.name: f.dtype() for f in longest}
    for f in shortest:
        name, dtype = f.name, f.dtype()
        dtype_longest = longest_map.setdefault(name, dtype)
        if dtype != dtype_longest:
            if supertype := get_supertype(dtype, dtype_longest, dtypes=dtypes):
                longest_map[name] = supertype
            else:
                return None
    return dtypes.Struct(longest_map)


def _struct_supertype(left: Struct, right: Struct, *, dtypes: DTypes) -> Struct | None:
    # https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L588-L603
    left_fields, right_fields = left.fields, right.fields
    if len(left_fields) != len(right_fields):
        return _struct_union_fields(left_fields, right_fields, dtypes=dtypes)
    new_fields = deque["Field"]()
    tp_field = dtypes.Field
    for a, b in zip(left_fields, right_fields):
        if a.name != b.name:
            return _struct_union_fields(left_fields, right_fields, dtypes=dtypes)
        if supertype := get_supertype(a.dtype(), b.dtype(), dtypes=dtypes):
            new_fields.append(tp_field(a.name, supertype))
        else:
            return None
    return dtypes.Struct(new_fields)


# TODO @dangotbanned: Change `dtypes: DTypes` -> `version: Version`
# - an `Enum` is safe to cache
# - we can split the `Version.V1` stuff into a different function
# - everything else can just use top-level imports from <equivalent to `polars.datatypes.classes`>
# - otherwise, only reference `version` for recursive calls
# TODO @dangotbanned: Initialize `base_types: frozenset[type[DType]` earlier
# - If we have `len(base_types) > 1`
#   - we can skip the first **7** branches
# - If we have `len(base_types) == 1`
#   - we can jump directly to the check for that `DType` and bail quickly
#   - skipping between **1-7** branches
def get_supertype(left: DType, right: DType, *, dtypes: DTypes) -> DType | None:  # noqa: C901, PLR0911, PLR0912
    """Given two data types, determine the data type that both types can reasonably safely be cast to.

    This function follows Polars' supertype rules:
    https://github.com/pola-rs/polars/blob/main/crates/polars-core/src/utils/supertype.rs

    Arguments:
        left: First data type.
        right: Second data type.
        dtypes: _description_

    Returns:
        The common supertype that both types can be safely cast to, or None if no such type exists.

    Examples:
        >>> import narwhals as nw
        >>> from narwhals.dtypes import get_supertype
        >>> get_supertype(nw.Int32(), nw.Int64())
        Int64
        >>> get_supertype(nw.Int32(), nw.Float64())
        Float64
        >>> get_supertype(nw.UInt8(), nw.Int8())
        Int16
        >>> get_supertype(nw.Date(), nw.Datetime("us"))
        Datetime(time_unit='us', time_zone=None)
        >>> get_supertype(nw.String(), nw.Int64()) is None
        True
    """
    if isinstance(left, dtypes.Datetime) and isinstance(right, dtypes.Datetime):
        if left.time_zone != right.time_zone:
            return None
        return dtypes.Datetime(
            _min_time_unit(left.time_unit, right.time_unit), left.time_zone
        )

    if isinstance(left, dtypes.Duration) and isinstance(right, dtypes.Duration):
        return dtypes.Duration(_min_time_unit(left.time_unit, right.time_unit))

    # For Enum types, categories **must** match
    if isinstance(left, dtypes.Enum) and isinstance(right, dtypes.Enum):
        return left if left.categories == right.categories else None

    if isinstance(left, dtypes.List) and isinstance(right, dtypes.List):
        if inner := get_supertype(left.inner(), right.inner(), dtypes=dtypes):
            return dtypes.List(inner)
        return None

    if isinstance(left, dtypes.Array) and isinstance(right, dtypes.Array):
        if (left.shape == right.shape) and (
            inner := get_supertype(left.inner(), right.inner(), dtypes=dtypes)
        ):
            return dtypes.Array(inner, left.size)
        return None

    # https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L496-L498
    if isinstance(left, dtypes.Struct) and isinstance(right, dtypes.Struct):
        return _struct_supertype(left, right, dtypes=dtypes)

    # NOTE: There are some other branches for `(Struct, DataType) -> Struct`
    # But, why?
    # https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L436-L442
    # https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L499-L507

    # TODO @dangotbanned: Find out why this isn't the first thing we do
    if left == right:
        return left

    # Numeric and Boolean -> Numeric
    if is_numeric(right) and is_boolean(left):
        return right
    if is_numeric(left) and is_boolean(right):
        return left

    # Both Integer
    if is_integer(left) and is_integer(right):
        return _integer_supertyping()(left, right)

    # Both Float
    if is_float(left) and is_float(right):
        return _max_bits(left, right)

    # Integer + Float -> Float
    #  * Small integers (Int8, Int16, UInt8, UInt16) + Float32 -> Float32
    #  * Larger integers (Int32+) + Float32 -> Float64
    #  * Any integer + Float64 -> Float64
    if is_integer(left) and is_float(right):
        if right._bits == 32 and left._bits <= 16:
            return dtypes.Float32()
        return dtypes.Float64()
    if is_float(left) and is_integer(right):
        if left._bits == 32 and right._bits <= 16:
            return dtypes.Float32()
        return dtypes.Float64()

    base_left, base_right = left.base_type(), right.base_type()

    # TODO @dangotbanned: Investigate using `frozenset((*left.__class__.__bases__, *right.__class__.__bases__))`
    # We can check for pairs like `combined_bases.issuperset((NumericType, TemporalType))`
    base_types = frozenset((base_left, base_right))

    # Date + Datetime -> Datetime
    if base_types == frozenset((dtypes.Date, dtypes.Datetime)):
        return left if isinstance(left, dtypes.Datetime) else right

    # Decimal with other numeric types
    # TODO @dangotbanned: Maybe branch off earlier if there is a numeric type?
    if Decimal in base_types and all(issubclass(tp, Numeric) for tp in base_types):
        return Decimal()

    # TODO @dangotbanned: (Date, {UInt,Int,Float}{32,64}) -> {Int,Float}{32,64}
    # https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L314-L328

    # TODO @dangotbanned: (Time, {Int,Float}{32,64}) -> {Int,Float}64
    # https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L369-L378

    # TODO @dangotbanned: (Datetime, {UInt,Int,Float}{32,64}) -> {Int,Float}64
    # https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L332-L345

    # TODO @dangotbanned: (Duration, {UInt,Int,Float}{32,64}) -> {Int,Float}64
    # https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L355-L367

    if String in base_types:
        # Categorical/Enum + String -> String
        if base_types.intersection((dtypes.Categorical, dtypes.Enum)):
            return String()
        # Every known type can be cast to a string except binary
        # https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L380-L382
        if Binary in base_types:
            return Binary()

    # TODO @dangotbanned: Maybe move this to the top?
    if Unknown in base_types:
        return Unknown()

    return None
