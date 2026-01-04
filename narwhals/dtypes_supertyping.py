# TODO @dangotbanned: Rename to `_supertyping`
# TODO @dangotbanned: Make `dtypes` a package
# TODO @dangotbanned: Move to `dtypes._supertyping`

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narwhals.dtypes import DType
    from narwhals.typing import DTypes, TimeUnit


@lru_cache(maxsize=4)
def _time_unit_to_index(time_unit: TimeUnit) -> int:
    """Convert time unit to an index for comparison (larger = more precise)."""
    return {"s": 0, "ms": 1, "us": 2, "ns": 3}[time_unit]


def _min_time_unit(a: TimeUnit, b: TimeUnit) -> TimeUnit:
    """Return the less precise time unit."""
    return a if _time_unit_to_index(a) <= _time_unit_to_index(b) else b


@lru_cache(4)
def _signed_int_to_bit_size(*, dtypes: DTypes) -> dict[DType, int]:
    """Mapping from signed integer types to their bit size."""
    return {
        dtypes.Int8(): 8,
        dtypes.Int16(): 16,
        dtypes.Int32(): 32,
        dtypes.Int64(): 64,
        dtypes.Int128(): 128,
    }


@lru_cache(4)
def _unsigned_int_to_bit_size(*, dtypes: DTypes) -> dict[DType, int]:
    """Mapping from bit size to signed integer type."""
    return {
        dtypes.UInt8(): 8,
        dtypes.UInt16(): 16,
        dtypes.UInt32(): 32,
        dtypes.UInt64(): 64,
        dtypes.UInt128(): 128,
    }


@lru_cache(4)
def _bit_size_to_signed_int(*, dtypes: DTypes) -> dict[int, DType]:
    """Mapping from bit size to signed integer type."""
    return {v: k for k, v in _signed_int_to_bit_size(dtypes=dtypes).items()}  # type: ignore[arg-type]


@lru_cache(4)
def _bit_size_to_unsigned_int(*, dtypes: DTypes) -> dict[int, DType]:
    """Mapping from bit size to unsigned integer type."""
    return {v: k for k, v in _unsigned_int_to_bit_size(dtypes=dtypes).items()}  # type: ignore[arg-type]


def _get_integer_supertype(left: DType, right: DType, *, dtypes: DTypes) -> DType | None:
    """Get supertype for two integer types.

    Following Polars rules:

    - Same signedness: return the larger type
    - Mixed signedness: promote to signed with enough bits to hold both
    - Int64 + UInt64 -> Float64 (following Polars)
    """
    left_signed = left.is_signed_integer()
    right_signed = right.is_signed_integer()

    signed_int_to_bit_size = _signed_int_to_bit_size(dtypes=dtypes)  # type: ignore[arg-type]
    unsigned_int_to_bit_size = _unsigned_int_to_bit_size(dtypes=dtypes)  # type: ignore[arg-type]

    left_bits = signed_int_to_bit_size.get(left) or unsigned_int_to_bit_size.get(left)
    right_bits = signed_int_to_bit_size.get(right) or unsigned_int_to_bit_size.get(right)

    if left_bits is None or right_bits is None:  # pragma: no cover
        return None

    # Same signedness: return larger type
    if left_signed == right_signed:
        max_bits = max(left_bits, right_bits)
        return (
            _bit_size_to_signed_int(dtypes=dtypes)[max_bits]  # type: ignore[arg-type]
            if left_signed
            else _bit_size_to_unsigned_int(dtypes=dtypes)[max_bits]  # type: ignore[arg-type]
        )

    # Mixed signedness: need signed type that can hold both
    # The unsigned type needs to fit in a signed type with more bits
    signed_bits, unsigned_bits = (
        (left_bits, right_bits) if left_signed else (right_bits, left_bits)
    )

    # If signed type is strictly larger than unsigned, it can hold both
    if signed_bits > unsigned_bits:
        return _bit_size_to_signed_int(dtypes=dtypes)[signed_bits]  # type: ignore[arg-type]

    # Otherwise, need to go to the next larger signed type
    # For Int64 + UInt64, Polars uses Float64 instead of Int128
    if unsigned_bits >= 64:
        return dtypes.Float64()

    # Find the smallest signed integer that can hold the unsigned value
    required_bits = unsigned_bits * 2
    for bits in (16, 32, 64):
        if bits >= required_bits:
            return _bit_size_to_signed_int(dtypes=dtypes)[bits]  # type: ignore[arg-type]

    # Fallback to Float64 if no integer type large enough
    return dtypes.Float64()


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
        if (
            inner_super_type := get_supertype(left_inner, right_inner, dtypes=dtypes)
        ) is None:
            return None
        return dtypes.Array(inner_super_type, left.size)

    if isinstance(left, dtypes.Struct) and isinstance(right, dtypes.Struct):
        # `left_fields, right_fields = left.fields, right.fields`
        msg = "TODO"
        raise NotImplementedError(msg)

    if left == right:
        return left

    # Numeric and Boolean -> Numeric
    if right.is_numeric() and left.is_boolean():
        return right
    if left.is_numeric() and right.is_boolean():
        return left

    # Both Integer
    if left.is_integer() and right.is_integer():
        return _get_integer_supertype(left, right, dtypes=dtypes)

    # Both Float
    if left.is_float() and right.is_float():
        return (
            dtypes.Float64()
            if (left == dtypes.Float64() or right == dtypes.Float64())
            else dtypes.Float32()
        )

    # Integer + Float -> Float
    #  * Small integers (Int8, Int16, UInt8, UInt16) + Float32 -> Float32
    #  * Larger integers (Int32+) + Float32 -> Float64
    #  * Any integer + Float64 -> Float64
    if left.is_integer() and right.is_float():
        if right == dtypes.Float64():
            return dtypes.Float64()

        # Float32 case
        left_bits = _signed_int_to_bit_size(dtypes=dtypes).get(  # type: ignore[arg-type]
            left
        ) or _unsigned_int_to_bit_size(dtypes=dtypes).get(left)  # type: ignore[arg-type]
        if left_bits is not None and left_bits <= 16:
            return dtypes.Float32()
        return dtypes.Float64()

    if right.is_integer() and left.is_float():
        if left == dtypes.Float64():
            return dtypes.Float64()

        # Float32 case
        right_bits = _signed_int_to_bit_size(dtypes=dtypes).get(  # type: ignore[arg-type]
            right
        ) or _unsigned_int_to_bit_size(dtypes=dtypes).get(right)  # type: ignore[arg-type]
        if right_bits is not None and right_bits <= 16:
            return dtypes.Float32()
        return dtypes.Float64()

    # Decimal with other numeric types
    if (isinstance(left, dtypes.Decimal) and right.is_numeric()) or (
        isinstance(right, dtypes.Decimal) and left.is_numeric()
    ):
        return dtypes.Decimal()

    # Date + Datetime -> Datetime
    if isinstance(left, dtypes.Date) and isinstance(right, dtypes.Datetime):
        return right
    if isinstance(right, dtypes.Date) and isinstance(left, dtypes.Datetime):
        return left

    # Categorical/Enum + String -> String
    if (
        isinstance(left, dtypes.String)
        and (isinstance(right, (dtypes.Categorical, dtypes.Enum)))
    ) or (
        isinstance(right, dtypes.String)
        and (isinstance(left, (dtypes.Categorical, dtypes.Enum)))
    ):
        return dtypes.String()

    if isinstance(left, dtypes.Unknown) or isinstance(right, dtypes.Unknown):
        return dtypes.Unknown()

    return None
