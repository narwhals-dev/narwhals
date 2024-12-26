from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence

from narwhals._expression_parsing import parse_into_exprs
from narwhals.utils import remove_prefix

if TYPE_CHECKING:
    from pyspark.sql import Column

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.typing import IntoSparkLikeExpr
    from narwhals.typing import CompliantExpr

POLARS_TO_PYSPARK_AGGREGATIONS = {
    "len": "count",
    "std": "stddev",
}


class DuckDBGroupBy:
    def __init__(
        self,
        df: SparkLikeLazyFrame,
        keys: list[str],
        drop_null_keys: bool,  # noqa: FBT001
    ) -> None:
        self._df = df
        self._keys = keys

    def agg(
        self,
        *aggs: IntoSparkLikeExpr,
        **named_aggs: IntoSparkLikeExpr,
    ) -> SparkLikeLazyFrame:
        exprs = parse_into_exprs(
            *aggs,
            namespace=self._df.__narwhals_namespace__(),
            **named_aggs,
        )
        output_names: list[str] = copy(self._keys)
        for expr in exprs:
            if expr._output_names is None:  # pragma: no cover
                msg = (
                    "Anonymous expressions are not supported in group_by.agg.\n"
                    "Instead of `nw.all()`, try using a named expression, such as "
                    "`nw.col('a', 'b')`\n"
                )
                raise ValueError(msg)

            output_names.extend(expr._output_names)

        return agg_duckdb(
            self._df,
            exprs,
            self._keys,
            self._from_native_frame,
        )

    def _from_native_frame(self, df: SparkLikeLazyFrame) -> SparkLikeLazyFrame:
        from narwhals._duckdb.dataframe import DuckDBInterchangeFrame

        return DuckDBInterchangeFrame(df, version=self._df._version)


def get_spark_function(function_name: str) -> Column:
    from duckdb import FunctionExpression

    return FunctionExpression(function_name)


def agg_duckdb(
    df,
    exprs: Sequence[CompliantExpr[Column]],
    keys: list[str],
    from_dataframe: Callable[[Any], SparkLikeLazyFrame],
) -> SparkLikeLazyFrame:
    agg_columns = [*keys, *(expr._call(df) for expr in exprs)]
    try:
        result_simple = df._native_frame.aggregate(agg_columns, group_expr=",".join(keys))
    except ValueError as exc:  # pragma: no cover
        msg = "Failed to aggregated - does your aggregation function return a scalar?"
        raise RuntimeError(msg) from exc
    return from_dataframe(result_simple)
