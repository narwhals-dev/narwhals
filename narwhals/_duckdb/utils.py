from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import duckdb

from narwhals.utils import import_dtypes_module
from narwhals.utils import isinstance_or_issubclass

if TYPE_CHECKING:
    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.expr import DuckDBExpr
    from narwhals.dtypes import DType
    from narwhals.utils import Version

col = duckdb.ColumnExpression
"""Alias for `duckdb.ColumnExpression`."""

lit = duckdb.ConstantExpression
"""Alias for `duckdb.ConstantExpression`."""

when = duckdb.CaseExpression
"""Alias for `duckdb.CaseExpression`."""


class WindowInputs:
    __slots__ = ("expr", "order_by", "partition_by")

    def __init__(
        self,
        expr: duckdb.Expression,
        partition_by: Sequence[str],
        order_by: Sequence[str],
    ) -> None:
        self.expr = expr
        self.partition_by = partition_by
        self.order_by = order_by


def concat_str(*exprs: duckdb.Expression, separator: str = "") -> duckdb.Expression:
    """Concatenate many strings, NULL inputs are skipped.

    Wraps [concat] and [concat_ws] `FunctionExpression`(s).

    Arguments:
        exprs: Native columns.
        separator: String that will be used to separate the values of each column.

    Returns:
        A new native expression.

    [concat]: https://duckdb.org/docs/stable/sql/functions/char.html#concatstring-
    [concat_ws]: https://duckdb.org/docs/stable/sql/functions/char.html#concat_wsseparator-string-
    """
    return (
        duckdb.FunctionExpression("concat_ws", lit(separator), *exprs)
        if separator
        else duckdb.FunctionExpression("concat", *exprs)
    )


def evaluate_exprs(
    df: DuckDBLazyFrame, /, *exprs: DuckDBExpr
) -> list[tuple[str, duckdb.Expression]]:
    native_results: list[tuple[str, duckdb.Expression]] = []
    for expr in exprs:
        native_series_list = expr._call(df)
        output_names = expr._evaluate_output_names(df)
        if expr._alias_output_names is not None:
            output_names = expr._alias_output_names(output_names)
        if len(output_names) != len(native_series_list):  # pragma: no cover
            msg = f"Internal error: got output names {output_names}, but only got {len(native_series_list)} results"
            raise AssertionError(msg)
        native_results.extend(zip(output_names, native_series_list))
    return native_results


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
    if duckdb_dtype == "TIMESTAMP WITH TIME ZONE":
        # TODO(marco): is UTC correct, or should we be getting the connection timezone?
        # https://github.com/narwhals-dev/narwhals/issues/2165
        return dtypes.Datetime(time_zone="UTC")
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
    if match_ := re.match(r"(\w+)((?:\[\d+\])+)", duckdb_dtype):
        duckdb_inner_type = match_.group(1)
        duckdb_shape = match_.group(2)
        shape = tuple(int(value) for value in re.findall(r"\[(\d+)\]", duckdb_shape))
        return dtypes.Array(
            inner=native_to_narwhals_dtype(duckdb_inner_type, version),
            shape=shape,
        )
    if duckdb_dtype.startswith("DECIMAL("):
        return dtypes.Decimal()
    if duckdb_dtype == "TIME":
        return dtypes.Time()
    if duckdb_dtype == "BLOB":
        return dtypes.Binary()
    return dtypes.Unknown()  # pragma: no cover


def narwhals_to_native_dtype(dtype: DType | type[DType], version: Version) -> str:
    dtypes = import_dtypes_module(version)
    if isinstance_or_issubclass(dtype, dtypes.Decimal):
        msg = "Casting to Decimal is not supported yet."
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Float64):
        return "DOUBLE"
    if isinstance_or_issubclass(dtype, dtypes.Float32):
        return "FLOAT"
    if isinstance_or_issubclass(dtype, dtypes.Int128):
        return "INT128"
    if isinstance_or_issubclass(dtype, dtypes.Int64):
        return "BIGINT"
    if isinstance_or_issubclass(dtype, dtypes.Int32):
        return "INTEGER"
    if isinstance_or_issubclass(dtype, dtypes.Int16):
        return "SMALLINT"
    if isinstance_or_issubclass(dtype, dtypes.Int8):
        return "TINYINT"
    if isinstance_or_issubclass(dtype, dtypes.UInt128):
        return "UINT128"
    if isinstance_or_issubclass(dtype, dtypes.UInt64):
        return "UBIGINT"
    if isinstance_or_issubclass(dtype, dtypes.UInt32):
        return "UINTEGER"
    if isinstance_or_issubclass(dtype, dtypes.UInt16):  # pragma: no cover
        return "USMALLINT"
    if isinstance_or_issubclass(dtype, dtypes.UInt8):  # pragma: no cover
        return "UTINYINT"
    if isinstance_or_issubclass(dtype, dtypes.String):
        return "VARCHAR"
    if isinstance_or_issubclass(dtype, dtypes.Boolean):  # pragma: no cover
        return "BOOLEAN"
    if isinstance_or_issubclass(dtype, dtypes.Time):
        return "TIME"
    if isinstance_or_issubclass(dtype, dtypes.Binary):
        return "BLOB"
    if isinstance_or_issubclass(dtype, dtypes.Categorical):
        msg = "Categorical not supported by DuckDB"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        _time_unit = dtype.time_unit
        _time_zone = dtype.time_zone
        msg = "todo"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Duration):  # pragma: no cover
        _time_unit = dtype.time_unit
        msg = "todo"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Date):  # pragma: no cover
        return "DATE"
    if isinstance_or_issubclass(dtype, dtypes.List):
        inner = narwhals_to_native_dtype(dtype.inner, version)
        return f"{inner}[]"
    if isinstance_or_issubclass(dtype, dtypes.Struct):  # pragma: no cover
        inner = ", ".join(
            f'"{field.name}" {narwhals_to_native_dtype(field.dtype, version)}'
            for field in dtype.fields
        )
        return f"STRUCT({inner})"
    if isinstance_or_issubclass(dtype, dtypes.Array):  # pragma: no cover
        shape = dtype.shape
        duckdb_shape_fmt = "".join(f"[{item}]" for item in shape)
        inner_dtype: Any = dtype
        for _ in shape:
            inner_dtype = inner_dtype.inner
        duckdb_inner = narwhals_to_native_dtype(inner_dtype, version)
        return f"{duckdb_inner}{duckdb_shape_fmt}"
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def generate_partition_by_sql(*partition_by: str) -> str:
    if not partition_by:
        return ""
    by_sql = ", ".join([f'"{x}"' for x in partition_by])
    return f"partition by {by_sql}"


def generate_order_by_sql(*order_by: str, ascending: bool) -> str:
    if ascending:
        by_sql = ", ".join([f'"{x}" asc nulls first' for x in order_by])
    else:
        by_sql = ", ".join([f'"{x}" desc nulls last' for x in order_by])
    return f"order by {by_sql}"


def ensure_type(obj: Any, *valid_types: type[Any]) -> None:
    # Use this for extra (possibly redundant) validation in places where we
    # use SQLExpression, as an extra guard against unsafe inputs.
    if not isinstance(obj, valid_types):  # pragma: no cover
        tp_names = " | ".join(tp.__name__ for tp in valid_types)
        msg = f"Expected {tp_names!r}, got: {type(obj).__name__!r}"
        raise TypeError(msg)
