from __future__ import annotations

from typing import Any

from narwhals import dtypes


def map_duckdb_dtype_to_narwhals_dtype(
    duckdb_dtype: Any,
) -> dtypes.DType:
    if duckdb_dtype == "BIGINT":
        return dtypes.Int64()
    if duckdb_dtype == "DOUBLE":
        return dtypes.Float64()
    if duckdb_dtype == "VARCHAR":
        return dtypes.String()
    msg = (  # pragma: no cover
        f"Invalid dtype, got: {duckdb_dtype}.\n\n"
        "If you believe this dtype should be supported in Narwhals, "
        "please report an issue at https://github.com/narwhals-dev/narwhals."
    )
    raise AssertionError(msg)


class DuckDBInterchangeFrame:
    def __init__(self, df: Any) -> None:
        self._native_frame = df

    def __narwhals_dataframe__(self) -> Any:
        return self

    def __getitem__(self, item: str) -> InterchangeSeries:
        from narwhals._duckdb.series import DuckDBInterchangeSeries

        return DuckDBInterchangeSeries(self._native_frame.select(item))

    @property
    def schema(self) -> dict[str, dtypes.DType]:
        return {
            column_name: map_duckdb_dtype_to_narwhals_dtype(
                self._native_frame.get_column_by_name(column_name).dtype
            )
            for column_name in zip(self._native_frame.columns, self._native_frame.types)
        }

    def __getattr__(self, attr: str) -> NoReturn:
        msg = (
            f"Attribute {attr} is not supported for metadata-only dataframes.\n\n"
            "Hint: you probably called `nw.from_native` on an object which isn't fully "
            "supported by Narwhals, yet implements `__dataframe__`. If you would like to "
            "see this kind of object supported in Narwhals, please open a feature request "
            "at https://github.com/narwhals-dev/narwhals/issues."
        )
        raise NotImplementedError(msg)
