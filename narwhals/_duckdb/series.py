from __future__ import annotations

from narwhals._duckdb.dataframe import map_duckdb_dtype_to_narwhals_dtype


class DuckDBInterchangeSeries:
    def __init__(self, df: Any) -> None:
        self._native_series = df

    def __narwhals_series__(self) -> Any:
        return self

    @property
    def dtype(self) -> dtypes.DType:
        return map_duckdb_dtype_to_narwhals_dtype(self._native_series.types[0])

    def __getattr__(self, attr: str) -> NoReturn:
        msg = (
            f"Attribute {attr} is not supported for metadata-only dataframes.\n\n"
            "Hint: you probably called `nw.from_native` on an object which isn't fully "
            "supported by Narwhals, yet implements `__dataframe__`. If you would like to "
            "see this kind of object supported in Narwhals, please open a feature request "
            "at https://github.com/narwhals-dev/narwhals/issues."
        )
        raise NotImplementedError(msg)
