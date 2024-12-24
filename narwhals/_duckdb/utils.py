from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals.exceptions import InvalidIntoExprError

if TYPE_CHECKING:
    import duckdb

    from narwhals._duckdb.dataframe import DuckDBInterchangeFrame
    from narwhals._duckdb.typing import IntoDuckDBExpr


def get_column_name(df: DuckDBInterchangeFrame, column: duckdb.Expression) -> str:
    return str(df._native_frame.select(column).columns[0])


def maybe_evaluate(df: DuckDBInterchangeFrame, obj: Any) -> Any:
    import duckdb

    from narwhals._duckdb.expr import DuckDBExpr

    if isinstance(obj, DuckDBExpr):
        column_results = obj._call(df)
        if len(column_results) != 1:  # pragma: no cover
            msg = "Multi-output expressions (e.g. `nw.all()` or `nw.col('a', 'b')`) not supported in this context"
            raise NotImplementedError(msg)
        column_result = column_results[0]
        if obj._returns_scalar:
            msg = "Reductions are not yet supported for DuckDB, at least until they implement duckdb.WindowExpression"
            raise NotImplementedError(msg)
        return column_result
    return duckdb.ConstantExpression(obj)


def parse_exprs_and_named_exprs(
    df: DuckDBInterchangeFrame,
    *exprs: IntoDuckDBExpr,
    **named_exprs: IntoDuckDBExpr,
) -> dict[str, duckdb.Expression]:
    result_columns: dict[str, list[duckdb.Expression]] = {}
    for expr in exprs:
        column_list = _columns_from_expr(df, expr)
        if isinstance(expr, str):  # pragma: no cover
            output_names = [expr]
        elif expr._output_names is None:
            output_names = [get_column_name(df, col) for col in column_list]
        else:
            output_names = expr._output_names
        result_columns.update(zip(output_names, column_list))
    for col_alias, expr in named_exprs.items():
        columns_list = _columns_from_expr(df, expr)
        if len(columns_list) != 1:  # pragma: no cover
            msg = "Named expressions must return a single column"
            raise AssertionError(msg)
        result_columns[col_alias] = columns_list[0]
    return result_columns


def _columns_from_expr(
    df: DuckDBInterchangeFrame, expr: IntoDuckDBExpr
) -> list[duckdb.Expression]:
    if isinstance(expr, str):  # pragma: no cover
        from duckdb import ColumnExpression

        return [ColumnExpression(expr)]
    elif hasattr(expr, "__narwhals_expr__"):
        col_output_list = expr._call(df)
        if expr._output_names is not None and (
            len(col_output_list) != len(expr._output_names)
        ):  # pragma: no cover
            msg = "Safety assertion failed, please report a bug to https://github.com/narwhals-dev/narwhals/issues"
            raise AssertionError(msg)
        return col_output_list
    else:
        raise InvalidIntoExprError.from_invalid_type(type(expr))
