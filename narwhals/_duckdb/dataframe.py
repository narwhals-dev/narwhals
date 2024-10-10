from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any

from narwhals.utils import parse_version

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._duckdb.series import DuckDBInterchangeSeries
    from narwhals.dtypes import DType
    from narwhals.typing import DTypes


def map_duckdb_dtype_to_narwhals_dtype(duckdb_dtype: Any, dtypes: DTypes) -> DType:
    duckdb_dtype = str(duckdb_dtype)
    if duckdb_dtype == "BIGINT":
        return dtypes.Int64()
    if duckdb_dtype == "INTEGER":
        return dtypes.Int32()
    if duckdb_dtype == "SMALLINT":
        return dtypes.Int16()
    if duckdb_dtype == "TINYINT":
        return dtypes.Int8()
    if duckdb_dtype == "UBIGINT":
        return dtypes.UInt64()
    if duckdb_dtype == "UINTEGER":
        return dtypes.UInt32()
    if duckdb_dtype == "USMALLINT":
        return dtypes.UInt16()
    if duckdb_dtype == "UTINYINT":
        return dtypes.UInt8()
    if duckdb_dtype == "DOUBLE":
        return dtypes.Float64()
    if duckdb_dtype == "FLOAT":
        return dtypes.Float32()
    if duckdb_dtype == "VARCHAR":
        return dtypes.String()
    if duckdb_dtype == "DATE":
        return dtypes.Date()
    if duckdb_dtype == "TIMESTAMP":
        return dtypes.Datetime()
    if duckdb_dtype == "BOOLEAN":
        return dtypes.Boolean()
    if duckdb_dtype == "INTERVAL":
        return dtypes.Duration()
    if duckdb_dtype.startswith("STRUCT"):
        return dtypes.Struct()
    if match_ := re.match(r"(.*)\[\]$", duckdb_dtype):
        return dtypes.List(map_duckdb_dtype_to_narwhals_dtype(match_.group(1), dtypes))
    if match_ := re.match(r"(\w+)\[(\d+)\]", duckdb_dtype):
        return dtypes.Array(
            map_duckdb_dtype_to_narwhals_dtype(match_.group(1), dtypes),
            int(match_.group(2)),
        )
    return dtypes.Unknown()


class DuckDBInterchangeFrame:
    def __init__(self, df: Any, dtypes: DTypes) -> None:
        self._native_frame = df
        self._dtypes = dtypes

    def __narwhals_dataframe__(self) -> Any:
        return self

    def __getitem__(self, item: str) -> DuckDBInterchangeSeries:
        from narwhals._duckdb.series import DuckDBInterchangeSeries

        return DuckDBInterchangeSeries(
            self._native_frame.select(item), dtypes=self._dtypes
        )

    def __getattr__(self, attr: str) -> Any:
        if attr == "schema":
            return {
                column_name: map_duckdb_dtype_to_narwhals_dtype(
                    duckdb_dtype, self._dtypes
                )
                for column_name, duckdb_dtype in zip(
                    self._native_frame.columns, self._native_frame.types
                )
            }

        msg = (  # pragma: no cover
            f"Attribute {attr} is not supported for metadata-only dataframes.\n\n"
            "If you would like to see this kind of object better supported in "
            "Narwhals, please open a feature request "
            "at https://github.com/narwhals-dev/narwhals/issues."
        )
        raise NotImplementedError(msg)  # pragma: no cover

    def to_pandas(self: Self) -> pd.DataFrame:
        import pandas as pd  # ignore-banned-import()

        if parse_version(pd.__version__) >= parse_version("1.0.0"):
            return self._native_frame.df()
        else:  # pragma: no cover
            msg = f"Conversion to pandas requires pandas>=1.0.0, found {pd.__version__}"
            raise NotImplementedError(msg)

    def to_arrow(self: Self) -> pa.Table:
        return self._native_frame.arrow()
