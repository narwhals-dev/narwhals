from __future__ import annotations

from typing import Any
from typing import Callable

from narwhals import dtypes
from narwhals.dependencies import get_pyarrow


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
        pyarrow_types.is_string_view: dtypes.String,
        pyarrow_types.is_date32: dtypes.Date,
        pyarrow_types.is_timestamp: dtypes.Datetime,
        pyarrow_types.is_duration: dtypes.Duration,
        pyarrow_types.is_dictionary: dtypes.Categorical,
    }

    for type_check, dtype_factory in dtype_mappers.items():
        if type_check(dtype):
            return dtype_factory()

    raise AssertionError
