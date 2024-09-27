from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals import dtypes

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._ibis.series import IbisInterchangeSeries


def map_ibis_dtype_to_narwhals_dtype(
    ibis_dtype: Any,
) -> dtypes.DType:
    if ibis_dtype.is_int64():
        return dtypes.Int64()
    if ibis_dtype.is_int32():
        return dtypes.Int32()
    if ibis_dtype.is_int16():
        return dtypes.Int16()
    if ibis_dtype.is_int8():
        return dtypes.Int8()
    if ibis_dtype.is_uint64():
        return dtypes.UInt64()
    if ibis_dtype.is_uint32():
        return dtypes.UInt32()
    if ibis_dtype.is_uint16():
        return dtypes.UInt16()
    if ibis_dtype.is_uint8():
        return dtypes.UInt8()
    if ibis_dtype.is_boolean():
        return dtypes.Boolean()
    if ibis_dtype.is_float64():
        return dtypes.Float64()
    if ibis_dtype.is_float32():
        return dtypes.Float32()
    if ibis_dtype.is_string():
        return dtypes.String()
    if ibis_dtype.is_date():
        return dtypes.Date()
    if ibis_dtype.is_timestamp():
        return dtypes.Datetime()
    if ibis_dtype.is_array():
        return dtypes.List()
    if ibis_dtype.is_struct():
        return dtypes.Struct()
    return dtypes.Unknown()  # pragma: no cover


class IbisInterchangeFrame:
    def __init__(self, df: Any) -> None:
        self._native_frame = df

    def __narwhals_dataframe__(self) -> Any:
        return self

    def __getitem__(self, item: str) -> IbisInterchangeSeries:
        from narwhals._ibis.series import IbisInterchangeSeries

        return IbisInterchangeSeries(self._native_frame[item])

    def to_pandas(self: Self) -> pd.DataFrame:
        return self._native_frame.to_pandas()

    def to_arrow(self: Self) -> pa.Table:
        return self._native_frame.to_pyarrow()

    def __getattr__(self, attr: str) -> Any:
        if attr == "schema":
            return {
                column_name: map_ibis_dtype_to_narwhals_dtype(ibis_dtype)
                for column_name, ibis_dtype in self._native_frame.schema().items()
            }
        msg = (
            f"Attribute {attr} is not supported for metadata-only dataframes.\n\n"
            "If you would like to see this kind of object better supported in "
            "Narwhals, please open a feature request "
            "at https://github.com/narwhals-dev/narwhals/issues."
        )
        raise NotImplementedError(msg)
