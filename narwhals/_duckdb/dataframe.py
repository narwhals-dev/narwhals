from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence

from narwhals._duckdb.utils import parse_exprs_and_named_exprs
from narwhals.dependencies import get_duckdb
from narwhals.utils import Implementation
from narwhals.utils import flatten
from narwhals.utils import import_dtypes_module
from narwhals.utils import parse_columns_to_drop
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from types import ModuleType

    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._duckdb.expr import DuckDBExpr
    from narwhals._duckdb.group_by import DuckDBGroupBy
    from narwhals._duckdb.namespace import DuckDBNamespace
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
        self._backend_version = (0, 0, 0)

    # This one is a historical mistake.
    # Keep around for backcompat, but remove in stable.v2
    def __narwhals_dataframe__(self) -> Any:
        return self

    def __narwhals_lazyframe__(self) -> Any:
        return self

    def __native_namespace__(self: Self) -> ModuleType:
        return get_duckdb()  # type: ignore[no-any-return]

    def __narwhals_namespace__(self) -> DuckDBNamespace:
        from narwhals._duckdb.namespace import DuckDBNamespace

        return DuckDBNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def __getitem__(self, item: str) -> DuckDBInterchangeSeries:
        from narwhals._duckdb.series import DuckDBInterchangeSeries

        return DuckDBInterchangeSeries(
            self._native_frame.select(item), version=self._version
        )

    def collect(self) -> Any:
        import pyarrow as pa  # ignore-banned-import()

        from narwhals._arrow.dataframe import ArrowDataFrame

        return ArrowDataFrame(
            native_dataframe=self._native_frame.arrow(),
            backend_version=parse_version(pa.__version__),
            version=self._version,
        )

    def head(self, n: int) -> Self:
        return self._from_native_frame(self._native_frame.limit(n))

    def select(
        self: Self,
        *exprs: Any,
        **named_exprs: Any,
    ) -> Self:
        new_columns_map = parse_exprs_and_named_exprs(self, *exprs, **named_exprs)
        if not new_columns_map:
            # TODO(marco): return empty relation with 0 columns?
            return self._from_native_frame(self._native_frame.limit(0))
        return self._from_native_frame(
            self._native_frame.select(
                *(val.alias(col) for col, val in new_columns_map.items())
            )
        )

    def drop(self: Self, columns: list[str], strict: bool) -> Self:  # noqa: FBT001
        columns_to_drop = parse_columns_to_drop(
            compliant_frame=self, columns=columns, strict=strict
        )
        selection = (col for col in self.columns if col not in columns_to_drop)
        return self._from_native_frame(self._native_frame.select(*selection))

    def lazy(self) -> Self:
        # TODO(marco): is this right? probably not
        return self

    def with_columns(
        self: Self,
        *exprs: Any,
        **named_exprs: Any,
    ) -> Self:
        from duckdb import ColumnExpression

        new_columns_map = parse_exprs_and_named_exprs(self, *exprs, **named_exprs)
        result = []
        for col in self._native_frame.columns:
            if col in new_columns_map:
                result.append(new_columns_map.pop(col).alias(col))
            else:
                result.append(ColumnExpression(col))
        for col, value in new_columns_map.items():
            result.append(value.alias(col))
        return self._from_native_frame(self._native_frame.select(*result))

    def filter(self, *predicates: DuckDBExpr) -> Self:
        from narwhals._duckdb.namespace import DuckDBNamespace

        if (
            len(predicates) == 1
            and isinstance(predicates[0], list)
            and all(isinstance(x, bool) for x in predicates[0])
        ):
            msg = "`LazyFrame.filter` is not supported for DuckDB backend with boolean masks."
            raise NotImplementedError(msg)
        plx = DuckDBNamespace(
            backend_version=self._backend_version, version=self._version
        )
        expr = plx.all_horizontal(*predicates)
        # Safety: all_horizontal's expression only returns a single column.
        condition = expr._call(self)[0]
        native_frame = self._native_frame.filter(condition)
        return self._from_native_frame(native_frame)

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
        elif attr == "_implementation":
            return Implementation.DUCKDB

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

    def group_by(self: Self, *keys: str, drop_null_keys: bool) -> DuckDBGroupBy:
        from narwhals._duckdb.group_by import DuckDBGroupBy

        return DuckDBGroupBy(
            compliant_frame=self, keys=list(keys), drop_null_keys=drop_null_keys
        )

    def join(
        self: Self,
        other: Self,
        *,
        how: Literal[left, inner, outer, cross, anti, semi] = "inner",
        left_on: str | list[str] | None,
        right_on: str | list[str] | None,
        suffix: str,
    ) -> Self:
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]
        if how != "inner":
            raise NotImplementedError("..")
        condition = ""
        for left, right in zip(left_on, right_on):
            condition += f"lhs.{left} = rhs.{right}"
        return self._from_native_frame(
            self._native_frame.set_alias("lhs").join(
                other._native_frame.set_alias("rhs"), condition=condition
            ),
        )

    def collect_schema(self) -> dict[str, DType]:
        return {
            column_name: native_to_narwhals_dtype(str(duckdb_dtype), self._version)
            for column_name, duckdb_dtype in zip(
                self._native_frame.columns, self._native_frame.types
            )
        }

    def sort(
        self: Self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool = False,
    ) -> Self:
        flat_by = flatten([*flatten([by]), *more_by])
        if isinstance(descending, bool):
            descending = [descending] * len(flat_by)
        descending_str = ["desc" if x else "" for x in descending]

        result = self._native_frame.order(
            ",".join(
                (
                    f"{col} {desc} nulls last"
                    if nulls_last
                    else f"{col} {desc} nulls first"
                    for col, desc in zip(flat_by, descending_str)
                )
            )
        )
        return self._from_native_frame(result)
