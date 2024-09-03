from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._pandas_like.utils import translate_dtype

if TYPE_CHECKING:
    from pyspark.pandas import Series
    from pyspark.sql import Column

    from narwhals._pyspark.dataframe import PySparkLazyFrame
    from narwhals._pyspark.typing import IntoPySparkExpr
    from narwhals.dtypes import DType


def translate_pandas_api_dtype(series: Series) -> DType:
    return translate_dtype(series)


def parse_exprs_and_named_exprs(
    df: PySparkLazyFrame, *exprs: IntoPySparkExpr, **named_exprs: IntoPySparkExpr
) -> list[Column]:
    from pyspark.sql import functions as F

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
