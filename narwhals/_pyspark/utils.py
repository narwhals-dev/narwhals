from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals import dtypes
from narwhals._pandas_like.utils import translate_dtype

if TYPE_CHECKING:
    from pyspark.pandas import Series
    from pyspark.sql import Column
    from pyspark.sql import types as pyspark_types

    from narwhals._pyspark.dataframe import PySparkLazyFrame
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
) -> list[Column]:
    from pyspark.sql import functions as F  # noqa: N812

    def _cols_from_expr(expr: IntoPySparkExpr) -> list[Column]:
        if isinstance(expr, str):
            return [F.col(expr)]
        elif hasattr(expr, "__narwhals_expr__"):
            return expr._call(df)
        else:  # pragma: no cover
            msg = f"Expected expression or column name, got: {expr}"
            raise TypeError(msg)

    columns_list = []
    for expr in exprs:
        pyspark_cols = _cols_from_expr(expr)
        columns_list.extend(pyspark_cols)

    for col_alias, expr in named_exprs.items():
        pyspark_cols = _cols_from_expr(expr)
        if len(pyspark_cols) != 1:  # pragma: no cover
            msg = "Named expressions must return a single column"
            raise AssertionError(msg)
        columns_list.extend([pyspark_cols[0].alias(col_alias)])
    return columns_list
