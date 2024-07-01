from __future__ import annotations

from typing import Any
from typing import Callable

from narwhals import dtypes
from narwhals.dependencies import get_pyarrow
from narwhals.utils import isinstance_or_issubclass


def translate_dtype(dtype: Any) -> dtypes.DType:
    pyarrow = get_pyarrow()
    pyarrow_types = pyarrow.types

    dtype_mappers: dict[Callable[[Any], bool], type[dtypes.DType]] = {
        pyarrow_types.is_int64: dtypes.Int64,
        pyarrow_types.is_int32: dtypes.Int32,
        pyarrow_types.is_int16: dtypes.Int16,
        pyarrow_types.is_int8: dtypes.Int8,
        pyarrow_types.is_uint64: dtypes.UInt64,
        pyarrow_types.is_uint32: dtypes.UInt32,
        pyarrow_types.is_uint16: dtypes.UInt16,
        pyarrow_types.is_uint8: dtypes.UInt8,
        pyarrow_types.is_boolean: dtypes.Boolean,
        pyarrow_types.is_float64: dtypes.Float64,
        pyarrow_types.is_float32: dtypes.Float32,
        pyarrow_types.is_string: dtypes.String,
        pyarrow_types.is_large_string: dtypes.String,
        getattr(pyarrow_types, "is_string_view", lambda _: False): dtypes.String,
        pyarrow_types.is_date32: dtypes.Date,
        pyarrow_types.is_timestamp: dtypes.Datetime,
        pyarrow_types.is_duration: dtypes.Duration,
        pyarrow_types.is_dictionary: dtypes.Categorical,
    }

    for type_check, dtype_factory in dtype_mappers.items():
        if type_check(dtype):
            return dtype_factory()

    raise AssertionError


def reverse_translate_dtype(dtype: dtypes.DType | type[dtypes.DType]) -> Any:
    from narwhals import dtypes

    pa = get_pyarrow()

    pyarrow_dtypes_mapping = {
        dtypes.Float64: pa.float64(),
        dtypes.Float32: pa.float32(),
        dtypes.Int64: pa.int64(),
        dtypes.Int32: pa.int32(),
        dtypes.Int16: pa.int16(),
        dtypes.Int8: pa.int8(),
        dtypes.UInt64: pa.uint64(),
        dtypes.UInt32: pa.uint32(),
        dtypes.UInt16: pa.uint16(),
        dtypes.UInt8: pa.uint8(),
        dtypes.String: pa.string(),
        dtypes.Boolean: pa.bool_(),
        # todo: what should the key be? let's keep it consistent
        # with Polars for now
        dtypes.Categorical: pa.dictionary(pa.uint32(), pa.string()),
        # Use Polars' default
        dtypes.Datetime: pa.timestamp("us"),
        # Use Polars' default
        dtypes.Duration: pa.duration("us"),
        dtypes.Date: pa.date32(),
    }

    for narwhals_dtype, pyarrow_dtype in pyarrow_dtypes_mapping.items():
        if isinstance_or_issubclass(dtype, narwhals_dtype):
            return pyarrow_dtype

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


def validate_dataframe_comparand(other: Any) -> Any:
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
            # broadcast
            msg = "not implemented yet"  # pragma: no cover
            raise NotImplementedError(msg)
        return other._native_series
    raise AssertionError("Please report a bug")
