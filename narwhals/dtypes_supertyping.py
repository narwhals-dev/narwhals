# TODO @dangotbanned: Rename to `_supertyping`
# TODO @dangotbanned: Make `dtypes` a package
# TODO @dangotbanned: Move to `dtypes._supertyping`

from __future__ import annotations

from functools import cache
from itertools import product
from operator import attrgetter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from typing_extensions import TypeAlias, TypeIs

    from narwhals.dtypes import (
        Boolean,
        DType,
        Float64,
        FloatType,
        IntegerType,
        NumericType,
        _Bits,
    )
    from narwhals.typing import DTypes, TimeUnit

    _HasBits: TypeAlias = "IntegerType | FloatType | type[IntegerType | FloatType]"

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
def _max_float(left: FloatType, right: FloatType) -> FloatType:
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
        i_bits, u_bits = signed._bits, unsigned._bits
        lookup_signed = reverse_lookup[SignedIntegerType]
        if i_bits > u_bits:
            return lookup_signed[i_bits]
        if u_bits in (8, 16, 32):  # noqa: PLR6201
            return lookup_signed[u_bits * 2]  # type: ignore[index]
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
            return None  # pragma: no cover
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

        # > Every known type can be cast to a string except binary
        # https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-core/src/utils/supertype.rs#L380-L382
        return dtypes.String()  # pragma: no cover

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
            return None  # pragma: no cover
        return dtypes.List(inner_super_type)

    if isinstance(left, dtypes.Array) and isinstance(right, dtypes.Array):
        if left.shape != right.shape:
            return None  # pragma: no cover
        left_inner, right_inner = left.inner, right.inner
        if isinstance(left_inner, type):
            left_inner = left_inner()
        if isinstance(right_inner, type):
            right_inner = right_inner()
        if (
            inner_super_type := get_supertype(left_inner, right_inner, dtypes=dtypes)
        ) is None:
            return None  # pragma: no cover
        return dtypes.Array(inner_super_type, left.size)

    if isinstance(left, dtypes.Struct) and isinstance(right, dtypes.Struct):
        # `left_fields, right_fields = left.fields, right.fields`
        msg = "TODO"
        raise NotImplementedError(msg)

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
        return _max_float(left, right)

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
    else:  # pragma: no cover  # noqa: RET505
        # NOTE: These don't have versioning, safe to use main
        from narwhals.dtypes import Decimal, NumericType, String, Unknown

        base_left, base_right = left.base_type(), right.base_type()
        base_types = frozenset((base_left, base_right))

        # Decimal with other numeric types
        # TODO @dangotbanned: Maybe branch off earlier if there is a numeric type?
        if Decimal in base_types and all(
            issubclass(tp, NumericType) for tp in base_types
        ):
            return Decimal()

        # Date + Datetime -> Datetime
        if base_types == frozenset((dtypes.Date, dtypes.Datetime)):
            return left if isinstance(left, dtypes.Datetime) else right

        # Categorical/Enum + String -> String
        if String in base_types and not base_types.isdisjoint(
            (dtypes.Categorical, dtypes.Enum)
        ):
            return String()

        # TODO @dangotbanned: Maybe move this to the top?
        if Unknown in base_types:
            return Unknown()

        return None
