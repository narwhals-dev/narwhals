# TODO @dangotbanned: Rename to `_supertyping`
# TODO @dangotbanned: Make `dtypes` a package
# TODO @dangotbanned: Move to `dtypes._supertyping`

from __future__ import annotations

from collections import deque
from itertools import product
from operator import attrgetter
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from typing_extensions import TypeAlias, TypeIs

    from narwhals.dtypes import (
        Boolean,
        DType,
        Field,
        Float64,
        FloatType,
        IntegerType,
        NumericType,
        SignedIntegerType,
        Struct,
        UnsignedIntegerType,
        _Bits,
    )
    from narwhals.typing import DTypes, IntoDType, TimeUnit

    _HasBits: TypeAlias = "IntegerType | FloatType | type[IntegerType | FloatType]"

    _Fn = TypeVar("_Fn", bound=Callable[..., Any])

    # NOTE: Hack to make `functools.cache` *not* negatively impact typing
    def cache(fn: _Fn, /) -> _Fn:
        return fn
else:
    from functools import cache


_SameNumericT = TypeVar(
    "_SameNumericT", "FloatType", "SignedIntegerType", "UnsignedIntegerType"
)
"""If both dtypes share one of these bases - pick the one with more bits."""

_TIME_UNIT_TO_INDEX: Mapping[TimeUnit, int] = {"s": 0, "ms": 1, "us": 2, "ns": 3}
"""Convert time unit to an index for comparison (larger = more precise)."""

_get_bits: Callable[[_HasBits], _Bits] = attrgetter("_bits")


# TODO @dangotbanned: Define the signatures inside `TYPE_CHECKING`,
# but implement using `operator.attrgetter` outside
def is_numeric(dtype: DType) -> TypeIs[NumericType]:
    return dtype.is_numeric()


def is_float(dtype: DType) -> TypeIs[FloatType]:
    return dtype.is_float()


def is_integer(dtype: DType) -> TypeIs[IntegerType]:
    return dtype.is_integer()


def is_boolean(dtype: DType) -> TypeIs[Boolean]:
    return dtype.is_boolean()


@cache
def _min_time_unit(a: TimeUnit, b: TimeUnit) -> TimeUnit:
    """Return the less precise time unit."""
    return min(a, b, key=_TIME_UNIT_TO_INDEX.__getitem__)


@cache
def _max_bits(left: _Bits, right: _Bits, /) -> _Bits:
    max_bits: _Bits = max(left, right)
    return max_bits


@cache
def _max_same_sign(left: _SameNumericT, right: _SameNumericT, /) -> _SameNumericT:
    return max(left, right, key=_get_bits)


@cache
def _integer_supertyping() -> Callable[[IntegerType, IntegerType], IntegerType | Float64]:
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
    from narwhals.dtypes import (
        Float64,
        IntegerType,
        SignedIntegerType,
        UnsignedIntegerType,
    )

    tps_int = SignedIntegerType.__subclasses__()
    tps_uint = UnsignedIntegerType.__subclasses__()

    reverse_lookup: Mapping[type[IntegerType], Mapping[_Bits, type[IntegerType]]] = {
        SignedIntegerType: {tp._bits: tp for tp in tps_int},
        UnsignedIntegerType: {tp._bits: tp for tp in tps_uint},
    }

    def value_same(
        left: type[IntegerType], right: type[IntegerType], /
    ) -> type[IntegerType]:
        return reverse_lookup[left.__base__ or IntegerType][
            _max_bits(left._bits, right._bits)
        ]

    def value_mixed(
        signed: type[IntegerType], unsigned: type[IntegerType], /
    ) -> type[IntegerType | Float64]:
        u_bits = unsigned._bits
        if signed._bits > u_bits:
            return signed
        if u_bits in (8, 16, 32):  # noqa: PLR6201
            return reverse_lookup[SignedIntegerType][u_bits * 2]  # type: ignore[index]
        return Float64

    lookup: Mapping[frozenset[type[IntegerType]], type[IntegerType | Float64]] = {
        frozenset((left, right)): fn_value(left, right)
        for iterable, fn_value in (
            (product(tps_int, tps_int), value_same),
            (product(tps_uint, tps_uint), value_same),
            (product(tps_int, tps_uint), value_mixed),
        )
        for left, right in iterable
    }

    def promote(left: IntegerType, right: IntegerType, /) -> IntegerType | Float64:
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
    longest_map = dict(_iter_init_fields(longest))
    for name, dtype in _iter_init_fields(shortest):
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
        a_dtype, b_dtype = _init_nested_pair(a.dtype, b.dtype)
        if supertype := get_supertype(a_dtype, b_dtype, dtypes=dtypes):
            new_fields.append(tp_field(a.name, supertype))
        else:
            return None
    return dtypes.Struct(new_fields)


def _iter_init_fields(fields: Iterable[Field]) -> Iterator[tuple[str, DType]]:
    # NOTE: This part would be easier if we did `IntoDType -> DType` **inside** `Field(...)`
    # Another option is defining `DType.__call__`, so `IntoDType()` becomes:
    #   - `DType(self).__call__(self) -> Self`
    #   - `NonNestedDType.__new__(NonNestedDType) -> NonNestedDType`
    from narwhals.dtypes import DTypeClass

    for f in fields:
        if isinstance(f.dtype, DTypeClass):
            yield f.name, f.dtype()
        else:
            yield f.name, f.dtype


def _init_nested_pair(left: IntoDType, right: IntoDType) -> tuple[DType, DType]:
    from narwhals.dtypes import DTypeClass

    # Handle case where inner is a type vs instance
    left_: DType = left() if isinstance(left, DTypeClass) else left
    right_: DType = right() if isinstance(right, DTypeClass) else right
    return left_, right_


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

    # For Enum types, categories must match
    if isinstance(left, dtypes.Enum) and isinstance(right, dtypes.Enum):
        if left.categories == right.categories:
            return left
        # TODO(FBruzzesi): Should we merge the categories? return dtypes.Enum((*left_cats, *right_cat))
        # NOTE: (@dangotbanned): Seems like something polars doesn't want https://github.com/pola-rs/polars/issues/22001
        # # https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/datatypes/dtype.rs#L1275-L1279
        return dtypes.String()

    if isinstance(left, dtypes.List) and isinstance(right, dtypes.List):
        left_inner, right_inner = left.inner, right.inner
        # Handle case where inner is a type vs instance
        if isinstance(left_inner, type):
            left_inner = left_inner()
        if isinstance(right_inner, type):
            right_inner = right_inner()
        if (
            inner_super_type := get_supertype(left_inner, right_inner, dtypes=dtypes)
        ) is None:
            return None
        return dtypes.List(inner_super_type)

    if isinstance(left, dtypes.Array) and isinstance(right, dtypes.Array):
        if left.shape != right.shape:
            return None
        left_inner, right_inner = left.inner, right.inner
        if isinstance(left_inner, type):
            left_inner = left_inner()
        if isinstance(right_inner, type):
            right_inner = right_inner()
        if inner_super_type := get_supertype(left_inner, right_inner, dtypes=dtypes):
            return dtypes.Array(inner_super_type, left.size)
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
        return _max_same_sign(left, right)

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

    # NOTE: These don't have versioning, safe to use main
    from narwhals.dtypes import Binary, Decimal, NumericType, String, Unknown

    base_left, base_right = left.base_type(), right.base_type()

    # TODO @dangotbanned: Investigate using `frozenset((*left.__class__.__bases__, *right.__class__.__bases__))`
    # We can check for pairs like `combined_bases.issuperset((NumericType, TemporalType))`
    base_types = frozenset((base_left, base_right))

    # Date + Datetime -> Datetime
    if base_types == frozenset((dtypes.Date, dtypes.Datetime)):
        return left if isinstance(left, dtypes.Datetime) else right

    # Decimal with other numeric types
    # TODO @dangotbanned: Maybe branch off earlier if there is a numeric type?
    if Decimal in base_types and all(issubclass(tp, NumericType) for tp in base_types):
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
