from __future__ import annotations

from typing import Any

from narwhals import dtypes


def map_ibis_dtype_to_narwhals_dtype(
    ibis_dtype: Any,
) -> dtypes.DType:
    import ibis

    if ibis_dtype == ibis.dtype("int64"):
        return dtypes.Int64()
    if ibis_dtype == ibis.dtype("int32"):
        return dtypes.Int32()
    if ibis_dtype == ibis.dtype("int16"):
        return dtypes.Int16()
    if ibis_dtype == ibis.dtype("int8"):
        return dtypes.Int8()
    if ibis_dtype == ibis.dtype("uint64"):
        return dtypes.UInt64()
    if ibis_dtype == ibis.dtype("uint32"):
        return dtypes.UInt32()
    if ibis_dtype == ibis.dtype("uint16"):
        return dtypes.UInt16()
    if ibis_dtype == ibis.dtype("uint8"):
        return dtypes.UInt8()
    if ibis_dtype == ibis.dtype("float64"):
        return dtypes.Float64()
    if ibis_dtype == ibis.dtype("float32"):
        return dtypes.Float32()
    msg = (  # pragma: no cover
        f"Invalid dtype, got: {ibis_dtype}.\n\n"
        "If you believe this dtype should be supported in Narwhals, "
        "please report an issue at https://github.com/narwhals-dev/narwhals."
    )
    raise AssertionError(msg)


class IbisInterchangeFrame:
    def __init__(self, df: Any) -> None:
        self._native_frame = df

    def __narwhals_dataframe__(self) -> Any:
        return self

    def __getitem__(self, item: str) -> InterchangeSeries:
        from narwhals._ibis.series import IbisInterchangeSeries

        return IbisInterchangeSeries(self._native_frame[item])

    def __getattr__(self, attr: str) -> NoReturn:
        if attr == "schema":
            return {
                column_name: map_ibis_dtype_to_narwhals_dtype(ibis_dtype)
                for column_name, ibis_dtype in self._native_frame.schema().items()
            }
        msg = (
            f"Attribute {attr} is not supported for metadata-only dataframes.\n\n"
            "Hint: you probably called `nw.from_native` on an object which isn't fully "
            "supported by Narwhals, yet implements `__dataframe__`. If you would like to "
            "see this kind of object supported in Narwhals, please open a feature request "
            "at https://github.com/narwhals-dev/narwhals/issues."
        )
        raise NotImplementedError(msg)
