from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any

from narwhals.dependencies import get_duckdb
from narwhals.utils import import_dtypes_module
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from types import ModuleType

    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._duckdb.series import DuckDBInterchangeSeries
    from narwhals.dtypes import DType
    from narwhals.utils import Version


@lru_cache(maxsize=16)
def native_to_narwhals_dtype(duckdb_dtype: str, version: Version) -> DType:
    dtypes = import_dtypes_module(version)
    if duckdb_dtype == "HUGEINT":
        return dtypes.Int128()
    if duckdb_dtype == "BIGINT":
        return dtypes.Int64()
    if duckdb_dtype == "INTEGER":
        return dtypes.Int32()
    if duckdb_dtype == "SMALLINT":
        return dtypes.Int16()
    if duckdb_dtype == "TINYINT":
        return dtypes.Int8()
    if duckdb_dtype == "UHUGEINT":
        return dtypes.UInt128()
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
        matchstruc_ = re.findall(r"(\w+)\s+(\w+)", duckdb_dtype)
        return dtypes.Struct(
            [
                dtypes.Field(
                    matchstruc_[i][0],
                    native_to_narwhals_dtype(matchstruc_[i][1], version),
                )
                for i in range(len(matchstruc_))
            ]
        )
    if match_ := re.match(r"(.*)\[\]$", duckdb_dtype):
        return dtypes.List(native_to_narwhals_dtype(match_.group(1), version))
    if match_ := re.match(r"(\w+)\[(\d+)\]", duckdb_dtype):
        return dtypes.Array(
            native_to_narwhals_dtype(match_.group(1), version),
            int(match_.group(2)),
        )
    if duckdb_dtype.startswith("DECIMAL("):
        return dtypes.Decimal()
    return dtypes.Unknown()  # pragma: no cover


class DuckDBInterchangeFrame:
    def __init__(self, df: Any, version: Version) -> None:
        self._native_frame = df
        self._version = version

    def __narwhals_dataframe__(self) -> Any:
        return self

    def __native_namespace__(self: Self) -> ModuleType:
        return get_duckdb()  # type: ignore[no-any-return]

    def __getitem__(self, item: str) -> DuckDBInterchangeSeries:
        from narwhals._duckdb.series import DuckDBInterchangeSeries

        return DuckDBInterchangeSeries(
            self._native_frame.select(item), version=self._version
        )

    def select(
        self: Self,
        *exprs: Any,
        **named_exprs: Any,
    ) -> Self:
        if named_exprs or not all(isinstance(x, str) for x in exprs):  # pragma: no cover
            msg = (
                "`select`-ing not by name is not supported for DuckDB backend.\n\n"
                "If you would like to see this kind of object better supported in "
                "Narwhals, please open a feature request "
                "at https://github.com/narwhals-dev/narwhals/issues."
            )
            raise NotImplementedError(msg)

        return self._from_native_frame(self._native_frame.select(*exprs))

    def __getattr__(self, attr: str) -> Any:
        if attr == "schema":
            return {
                column_name: native_to_narwhals_dtype(str(duckdb_dtype), self._version)
                for column_name, duckdb_dtype in zip(
                    self._native_frame.columns, self._native_frame.types
                )
            }
        elif attr == "columns":
            return self._native_frame.columns

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

    def _change_version(self: Self, version: Version) -> Self:
        return self.__class__(self._native_frame, version=version)

    def _from_native_frame(self: Self, df: Any) -> Self:
        return self.__class__(df, version=self._version)
