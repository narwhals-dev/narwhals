from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import duckdb

from narwhals.utils import Version
from narwhals.utils import isinstance_or_issubclass

if TYPE_CHECKING:
    from duckdb.typing import DuckDBPyType

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.expr import DuckDBExpr
    from narwhals.dtypes import DType

UNITS_DICT = {
    "y": "year",
    "q": "quarter",
    "mo": "month",
    "d": "day",
    "h": "hour",
    "m": "minute",
    "s": "second",
    "ms": "millisecond",
    "us": "microsecond",
    "ns": "nanosecond",
}

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


class UnorderableWindowInputs:
    __slots__ = ("expr", "partition_by")

    def __init__(
        self,
        expr: duckdb.Expression,
        partition_by: Sequence[str],
    ) -> None:
        self.expr = expr
        self.partition_by = partition_by


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


def native_to_narwhals_dtype(duckdb_dtype: DuckDBPyType, version: Version) -> DType:
    duckdb_dtype_id = duckdb_dtype.id
    dtypes = version.dtypes

    # Handle nested data types first
    if duckdb_dtype_id == "list":
        return dtypes.List(native_to_narwhals_dtype(duckdb_dtype.child, version=version))

    if duckdb_dtype_id == "struct":
        children = duckdb_dtype.children
        return dtypes.Struct(
            [
                dtypes.Field(
                    name=child[0],
                    dtype=native_to_narwhals_dtype(child[1], version=version),
                )
                for child in children
            ]
        )

    if duckdb_dtype_id == "array":
        child, size = duckdb_dtype.children
        shape: list[int] = [size[1]]

        while child[1].id == "array":
            child, size = child[1].children
            shape.insert(0, size[1])

        inner = native_to_narwhals_dtype(child[1], version=version)
        return dtypes.Array(inner=inner, shape=tuple(shape))

    if duckdb_dtype_id == "enum":
        if version is Version.V1:
            return dtypes.Enum()  # type: ignore[call-arg]
        categories = duckdb_dtype.children[0][1]
        return dtypes.Enum(categories=categories)

    return _non_nested_native_to_narwhals_dtype(duckdb_dtype_id, version)


@lru_cache(maxsize=16)
def _non_nested_native_to_narwhals_dtype(duckdb_dtype_id: str, version: Version) -> DType:
    dtypes = version.dtypes
    return {
        "hugeint": dtypes.Int128(),
        "bigint": dtypes.Int64(),
        "integer": dtypes.Int32(),
        "smallint": dtypes.Int16(),
        "tinyint": dtypes.Int8(),
        "uhugeint": dtypes.UInt128(),
        "ubigint": dtypes.UInt64(),
        "uinteger": dtypes.UInt32(),
        "usmallint": dtypes.UInt16(),
        "utinyint": dtypes.UInt8(),
        "double": dtypes.Float64(),
        "float": dtypes.Float32(),
        "varchar": dtypes.String(),
        "date": dtypes.Date(),
        "timestamp": dtypes.Datetime(),
        # TODO(marco): is UTC correct, or should we be getting the connection timezone?
        # https://github.com/narwhals-dev/narwhals/issues/2165
        "timestamp with time zone": dtypes.Datetime(time_zone="UTC"),
        "boolean": dtypes.Boolean(),
        "interval": dtypes.Duration(),
        "decimal": dtypes.Decimal(),
        "time": dtypes.Time(),
        "blob": dtypes.Binary(),
    }.get(duckdb_dtype_id, dtypes.Unknown())


def narwhals_to_native_dtype(dtype: DType | type[DType], version: Version) -> str:  # noqa: C901, PLR0912, PLR0915
    dtypes = version.dtypes
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
    if isinstance_or_issubclass(dtype, dtypes.Enum):
        if version is Version.V1:
            msg = "Converting to Enum is not supported in narwhals.stable.v1"
            raise NotImplementedError(msg)
        if isinstance(dtype, dtypes.Enum):
            categories = "'" + "', '".join(dtype.categories) + "'"
            return f"ENUM ({categories})"
        msg = "Can not cast / initialize Enum without categories present"
        raise ValueError(msg)

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


def generate_partition_by_sql(*partition_by: str | duckdb.Expression) -> str:
    if not partition_by:
        return ""
    by_sql = ", ".join([f"{col(x) if isinstance(x, str) else x}" for x in partition_by])
    return f"partition by {by_sql}"


def generate_order_by_sql(*order_by: str, ascending: bool) -> str:
    if ascending:
        by_sql = ", ".join([f"{col(x)} asc nulls first" for x in order_by])
    else:
        by_sql = ", ".join([f"{col(x)} desc nulls last" for x in order_by])
    return f"order by {by_sql}"


def ensure_type(obj: Any, *valid_types: type[Any]) -> None:
    # Use this for extra (possibly redundant) validation in places where we
    # use SQLExpression, as an extra guard against unsafe inputs.
    if not isinstance(obj, valid_types):  # pragma: no cover
        tp_names = " | ".join(tp.__name__ for tp in valid_types)
        msg = f"Expected {tp_names!r}, got: {type(obj).__name__!r}"
        raise TypeError(msg)
