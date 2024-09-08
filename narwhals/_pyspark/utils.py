from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals import dtypes
from narwhals._pandas_like.utils import translate_dtype

if TYPE_CHECKING:
    from pyspark.pandas import Series
    from pyspark.sql import types as pyspark_types

    from narwhals._pyspark.dataframe import PySparkLazyFrame
    from narwhals._pyspark.series import PySparkSeries
    from narwhals._pyspark.typing import IntoPySparkExpr


def translate_pandas_api_dtype(series: Series) -> dtypes.DType:
    return translate_dtype(series)


def translate_sql_api_dtype(dtype: pyspark_types.DataType) -> dtypes.DType:
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


def parse_exprs_and_named_exprs(
    df: PySparkLazyFrame, *exprs: IntoPySparkExpr, **named_exprs: IntoPySparkExpr
) -> dict[str, PySparkSeries]:
    def _series_from_expr(expr: IntoPySparkExpr) -> list[PySparkSeries]:
        if isinstance(expr, str):
            from narwhals._pyspark.series import PySparkSeries

            return [PySparkSeries(native_series=df._native_frame.select(expr), name=expr)]
        elif hasattr(expr, "__narwhals_expr__"):
            return expr._call(df)
        else:  # pragma: no cover
            msg = f"Expected expression or column name, got: {expr}"
            raise TypeError(msg)

    result_series = {}
    for expr in exprs:
        series_list = _series_from_expr(expr)
        for series in series_list:
            result_series[series.name] = series

    for col_alias, expr in named_exprs.items():
        series_list = _series_from_expr(expr)
        if len(series_list) != 1:  # pragma: no cover
            msg = "Named expressions must return a single column"
            raise AssertionError(msg)
        result_series[col_alias] = series_list[0]
    return result_series


def maybe_evaluate(df: PySparkLazyFrame, obj: Any) -> Any:
    from narwhals._pyspark.expr import PySparkExpr

    if isinstance(obj, PySparkExpr):
        series_result = obj._call(df)
        if len(series_result) != 1:  # pragma: no cover
            msg = "Multi-output expressions not supported in this context"
            raise NotImplementedError(msg)
        series = series_result[0]
        if obj._returns_scalar:
            raise NotImplementedError
        return series
    return obj


def validate_column_comparand(other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.

    If RHS is length 1, return the scalar value, so that the underlying
    library can broadcast it.
    """
    from narwhals._pyspark.dataframe import PySparkLazyFrame
    from narwhals._pyspark.series import PySparkSeries

    if isinstance(other, list):
        if len(other) > 1:
            # e.g. `plx.all() + plx.all()`
            msg = "Multi-output expressions are not supported in this context"
            raise ValueError(msg)
        other = other[0]
    if isinstance(other, PySparkLazyFrame):
        return NotImplemented
    if isinstance(other, PySparkSeries):
        if len(other) == 1:
            # broadcast
            return other[0]
        return other._native_column
    return other
