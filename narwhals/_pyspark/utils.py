from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals import dtypes

if TYPE_CHECKING:
    from pyspark.sql import Column
    from pyspark.sql import types as pyspark_types

    from narwhals._pyspark.dataframe import PySparkLazyFrame
    from narwhals._pyspark.typing import IntoPySparkExpr


def translate_sql_api_dtype(
    dtype: pyspark_types.DataType,
) -> dtypes.DType:  # pragma: no cover
    from pyspark.sql import types as pyspark_types

    if isinstance(dtype, pyspark_types.DoubleType):
        return dtypes.Float64()
    if isinstance(dtype, pyspark_types.FloatType):
        return dtypes.Float32()
    if isinstance(dtype, pyspark_types.LongType):
        return dtypes.Int64()
    if isinstance(dtype, pyspark_types.IntegerType):
        return dtypes.Int32()
    if isinstance(dtype, pyspark_types.ShortType):
        return dtypes.Int16()
    if isinstance(dtype, pyspark_types.ByteType):
        return dtypes.Int8()
    if isinstance(dtype, pyspark_types.DecimalType):
        return dtypes.Int32()
    string_types = [
        pyspark_types.StringType,
        pyspark_types.VarcharType,
        pyspark_types.CharType,
    ]
    if any(isinstance(dtype, t) for t in string_types):
        return dtypes.String()
    if isinstance(dtype, pyspark_types.BooleanType):
        return dtypes.Boolean()
    if isinstance(dtype, pyspark_types.DateType):
        return dtypes.Date()
    datetime_types = [
        pyspark_types.TimestampType,
        pyspark_types.TimestampNTZType,
    ]
    if any(isinstance(dtype, t) for t in datetime_types):
        return dtypes.Datetime()
    return dtypes.Unknown()


def get_column_name(df: PySparkLazyFrame, column: Column) -> str:
    return str(df._native_frame.select(column).columns[0])


def parse_exprs_and_named_exprs(
    df: PySparkLazyFrame, *exprs: IntoPySparkExpr, **named_exprs: IntoPySparkExpr
) -> dict[str, Column]:
    def _columns_from_expr(expr: IntoPySparkExpr) -> list[Column]:
        if isinstance(expr, str):  # pragma: no cover
            from pyspark.sql import functions as F  # noqa: N812

            return [F.col(expr)]
        elif hasattr(expr, "__narwhals_expr__"):
            col_output_list = expr._call(df)
            if expr._output_names is not None and (
                len(col_output_list) != len(expr._output_names)
            ):  # pragma: no cover
                msg = "Safety assertion failed, please report a bug to https://github.com/narwhals-dev/narwhals/issues"
                raise AssertionError(msg)
            return expr._call(df)
        else:  # pragma: no cover
            msg = f"Expected expression or column name, got: {expr}"
            raise TypeError(msg)

    result_columns: dict[str, list[Column]] = {}
    for expr in exprs:
        column_list = _columns_from_expr(expr)
        if isinstance(expr, str):  # pragma: no cover
            output_names = [expr]
        elif expr._output_names is None:
            output_names = [get_column_name(df, col) for col in column_list]
        else:
            output_names = expr._output_names
        result_columns.update(zip(output_names, column_list))
    for col_alias, expr in named_exprs.items():
        columns_list = _columns_from_expr(expr)
        if len(columns_list) != 1:  # pragma: no cover
            msg = "Named expressions must return a single column"
            raise AssertionError(msg)
        result_columns[col_alias] = columns_list[0]
    return result_columns


def maybe_evaluate(df: PySparkLazyFrame, obj: Any) -> Any:
    from narwhals._pyspark.expr import PySparkExpr

    if isinstance(obj, PySparkExpr):
        column_results = obj._call(df)
        if len(column_results) != 1:  # pragma: no cover
            msg = "Multi-output expressions not supported in this context"
            raise NotImplementedError(msg)
        column_result = column_results[0]
        if obj._returns_scalar:
            # Return scalar, let PySpark do its broadcasting
            from pyspark.sql import functions as F  # noqa: N812
            from pyspark.sql.window import Window

            return column_result.over(Window.partitionBy(F.lit(1)))
        return column_result
    return obj
