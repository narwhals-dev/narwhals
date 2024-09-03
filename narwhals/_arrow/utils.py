from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals import dtypes
from narwhals.utils import isinstance_or_issubclass

if TYPE_CHECKING:
    from narwhals._arrow.series import ArrowSeries


def translate_dtype(dtype: Any) -> dtypes.DType:
    import pyarrow as pa  # ignore-banned-import()

    if pa.types.is_int64(dtype):
        return dtypes.Int64()
    if pa.types.is_int32(dtype):
        return dtypes.Int32()
    if pa.types.is_int16(dtype):
        return dtypes.Int16()
    if pa.types.is_int8(dtype):
        return dtypes.Int8()
    if pa.types.is_uint64(dtype):
        return dtypes.UInt64()
    if pa.types.is_uint32(dtype):
        return dtypes.UInt32()
    if pa.types.is_uint16(dtype):
        return dtypes.UInt16()
    if pa.types.is_uint8(dtype):
        return dtypes.UInt8()
    if pa.types.is_boolean(dtype):
        return dtypes.Boolean()
    if pa.types.is_float64(dtype):
        return dtypes.Float64()
    if pa.types.is_float32(dtype):
        return dtypes.Float32()
    # bug in coverage? it shows `31->exit` (where `31` is currently the line number of
    # the next line), even though both when the if condition is true and false are covered
    if (  # pragma: no cover
        pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or getattr(pa.types, "is_string_view", lambda _: False)(dtype)
    ):
        return dtypes.String()
    if pa.types.is_date32(dtype):
        return dtypes.Date()
    if pa.types.is_timestamp(dtype):
        return dtypes.Datetime()
    if pa.types.is_duration(dtype):
        return dtypes.Duration()
    if pa.types.is_dictionary(dtype):
        return dtypes.Categorical()
    return dtypes.Unknown()  # pragma: no cover


def narwhals_to_native_dtype(dtype: dtypes.DType | type[dtypes.DType]) -> Any:
    import pyarrow as pa  # ignore-banned-import()

    from narwhals import dtypes

    if isinstance_or_issubclass(dtype, dtypes.Float64):
        return pa.float64()
    if isinstance_or_issubclass(dtype, dtypes.Float32):
        return pa.float32()
    if isinstance_or_issubclass(dtype, dtypes.Int64):
        return pa.int64()
    if isinstance_or_issubclass(dtype, dtypes.Int32):
        return pa.int32()
    if isinstance_or_issubclass(dtype, dtypes.Int16):
        return pa.int16()
    if isinstance_or_issubclass(dtype, dtypes.Int8):
        return pa.int8()
    if isinstance_or_issubclass(dtype, dtypes.UInt64):
        return pa.uint64()
    if isinstance_or_issubclass(dtype, dtypes.UInt32):
        return pa.uint32()
    if isinstance_or_issubclass(dtype, dtypes.UInt16):
        return pa.uint16()
    if isinstance_or_issubclass(dtype, dtypes.UInt8):
        return pa.uint8()
    if isinstance_or_issubclass(dtype, dtypes.String):
        return pa.string()
    if isinstance_or_issubclass(dtype, dtypes.Boolean):
        return pa.bool_()
    if isinstance_or_issubclass(dtype, dtypes.Categorical):
        # TODO(Unassigned): what should the key be? let's keep it consistent
        # with Polars for now
        return pa.dictionary(pa.uint32(), pa.string())
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        # Use Polars' default
        return pa.timestamp("us")
    if isinstance_or_issubclass(dtype, dtypes.Duration):
        # Use Polars' default
        return pa.duration("us")
    if isinstance_or_issubclass(dtype, dtypes.Date):
        return pa.date32()
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def validate_column_comparand(other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.

    If RHS is length 1, return the scalar value, so that the underlying
    library can broadcast it.
    """
    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.series import ArrowSeries

    if isinstance(other, list):
        if len(other) > 1:
            # e.g. `plx.all() + plx.all()`
            msg = "Multi-output expressions are not supported in this context"
            raise ValueError(msg)
        other = other[0]
    if isinstance(other, ArrowDataFrame):
        return NotImplemented
    if isinstance(other, ArrowSeries):
        if len(other) == 1:
            # broadcast
            return other[0]
        return other._native_series
    return other


def validate_dataframe_comparand(
    length: int, other: Any, backend_version: tuple[int, ...]
) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.
    """
    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.series import ArrowSeries

    if isinstance(other, ArrowDataFrame):
        return NotImplemented
    if isinstance(other, ArrowSeries):
        if len(other) == 1:
            import pyarrow as pa  # ignore-banned-import()

            value = other.item()
            if backend_version < (13,) and hasattr(value, "as_py"):  # pragma: no cover
                value = value.as_py()
            return pa.chunked_array([[value] * length])
        return other._native_series
    msg = "Please report a bug"  # pragma: no cover
    raise AssertionError(msg)


def horizontal_concat(dfs: list[Any]) -> Any:
    """
    Concatenate (native) DataFrames horizontally.

    Should be in namespace.
    """
    import pyarrow as pa  # ignore-banned-import()

    if not dfs:
        msg = "No dataframes to concatenate"  # pragma: no cover
        raise AssertionError(msg)

    names = [name for df in dfs for name in df.column_names]

    if len(set(names)) < len(names):  # pragma: no cover
        msg = "Expected unique column names"
        raise ValueError(msg)

    arrays = [a for df in dfs for a in df]
    return pa.Table.from_arrays(arrays, names=names)


def vertical_concat(dfs: list[Any]) -> Any:
    """
    Concatenate (native) DataFrames vertically.

    Should be in namespace.
    """
    if not dfs:
        msg = "No dataframes to concatenate"  # pragma: no cover
        raise AssertionError(msg)

    cols = set(dfs[0].column_names)
    for df in dfs:
        cols_current = set(df.column_names)
        if cols_current != cols:
            msg = "unable to vstack, column names don't match"
            raise TypeError(msg)

    import pyarrow as pa  # ignore-banned-import()

    return pa.concat_tables(dfs).combine_chunks()


def floordiv_compat(left: Any, right: Any) -> Any:
    # The following lines are adapted from pandas' pyarrow implementation.
    # Ref: https://github.com/pandas-dev/pandas/blob/262fcfbffcee5c3116e86a951d8b693f90411e68/pandas/core/arrays/arrow/array.py#L124-L154
    import pyarrow as pa  # ignore-banned-import()
    import pyarrow.compute as pc  # ignore-banned-import()

    if isinstance(left, (int, float)):
        left = pa.scalar(left)

    if isinstance(right, (int, float)):
        right = pa.scalar(right)

    if pa.types.is_integer(left.type) and pa.types.is_integer(right.type):
        divided = pc.divide_checked(left, right)
        if pa.types.is_signed_integer(divided.type):
            # GH 56676
            has_remainder = pc.not_equal(pc.multiply(divided, right), left)
            has_one_negative_operand = pc.less(
                pc.bit_wise_xor(left, right),
                pa.scalar(0, type=divided.type),
            )
            result = pc.if_else(
                pc.and_(
                    has_remainder,
                    has_one_negative_operand,
                ),
                # GH: 55561 ruff: ignore
                pc.subtract(divided, pa.scalar(1, type=divided.type)),
                divided,
            )
        else:
            result = divided  # pragma: no cover
        result = result.cast(left.type)
    else:
        divided = pc.divide(left, right)
        result = pc.floor(divided)
    return result


def cast_for_truediv(arrow_array: Any, pa_object: Any) -> tuple[Any, Any]:
    # Lifted from:
    # https://github.com/pandas-dev/pandas/blob/262fcfbffcee5c3116e86a951d8b693f90411e68/pandas/core/arrays/arrow/array.py#L108-L122
    import pyarrow as pa  # ignore-banned-import()
    import pyarrow.compute as pc  # ignore-banned-import()

    # Ensure int / int -> float mirroring Python/Numpy behavior
    # as pc.divide_checked(int, int) -> int
    if pa.types.is_integer(arrow_array.type) and pa.types.is_integer(pa_object.type):
        # GH: 56645.  # noqa: ERA001
        # https://github.com/apache/arrow/issues/35563
        return pc.cast(arrow_array, pa.float64(), safe=False), pc.cast(
            pa_object, pa.float64(), safe=False
        )

    return arrow_array, pa_object


def broadcast_series(series: list[ArrowSeries]) -> list[Any]:
    lengths = [len(s) for s in series]
    max_length = max(lengths)
    fast_path = all(_len == max_length for _len in lengths)

    if fast_path:
        return [s._native_series for s in series]

    import pyarrow as pa  # ignore-banned-import()

    reshaped = []
    for s, length in zip(series, lengths):
        s_native = s._native_series
        if max_length > 1 and length == 1:
            value = s_native[0]
            if s._backend_version < (13,) and hasattr(value, "as_py"):  # pragma: no cover
                value = value.as_py()
            reshaped.append(pa.array([value] * max_length, type=s_native.type))
        else:
            reshaped.append(s_native)

    return reshaped
