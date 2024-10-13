from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from narwhals._expression_parsing import is_simple_aggregation
from narwhals._expression_parsing import parse_into_exprs
from narwhals.utils import remove_prefix

if TYPE_CHECKING:
    from pyspark.sql import Column
    from pyspark.sql import GroupedData

    from narwhals._pyspark.dataframe import PySparkLazyFrame
    from narwhals._pyspark.expr import PySparkExpr
    from narwhals._pyspark.typing import IntoPySparkExpr

POLARS_TO_PYSPARK_AGGREGATIONS = {
    "len": "count",
}


class PySparkLazyGroupBy:
    def __init__(self, df: PySparkLazyFrame, keys: list[str]) -> None:
        self._df = df
        self._keys = keys
        self._grouped = self._df._native_frame.groupBy(*self._keys)

    def agg(
        self,
        *aggs: IntoPySparkExpr,
        **named_aggs: IntoPySparkExpr,
    ) -> PySparkLazyFrame:
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

        return agg_pyspark(
            self._grouped,
            exprs,
            self._keys,
            self._from_native_frame,
        )

    def _from_native_frame(self, df: PySparkLazyFrame) -> PySparkLazyFrame:
        from narwhals._pyspark.dataframe import PySparkLazyFrame

        return PySparkLazyFrame(
            df, backend_version=self._df._backend_version, dtypes=self._df._dtypes
        )


def get_spark_function(function_name: str) -> Column:
    from pyspark.sql import functions as F  # noqa: N812

    return getattr(F, function_name)


def agg_pyspark(
    grouped: GroupedData,
    exprs: list[PySparkExpr],
    keys: list[str],
    from_dataframe: Callable[[Any], PySparkLazyFrame],
) -> PySparkLazyFrame:
    for expr in exprs:
        if not is_simple_aggregation(expr):  # pragma: no cover
            msg = (
                "Non-trivial complex found.\n\n"
                "Hint: you were probably trying to apply a non-elementary aggregation with a "
                "dask dataframe.\n"
                "Please rewrite your query such that group-by aggregations "
                "are elementary. For example, instead of:\n\n"
                "    df.group_by('a').agg(nw.col('b').round(2).mean())\n\n"
                "use:\n\n"
                "    df.with_columns(nw.col('b').round(2)).group_by('a').agg(nw.col('b').mean())\n\n"
            )
            raise ValueError(msg)

    simple_aggregations: dict[str, Column] = {}
    for expr in exprs:
        if expr._depth == 0:  # pragma: no cover
            # e.g. agg(nw.len()) # noqa: ERA001
            if expr._output_names is None:  # pragma: no cover
                msg = "Safety assertion failed, please report a bug to https://github.com/narwhals-dev/narwhals/issues"
                raise AssertionError(msg)

            function_name = POLARS_TO_PYSPARK_AGGREGATIONS.get(
                expr._function_name, expr._function_name
            )
            for output_name in expr._output_names:
                agg_func = get_spark_function(function_name)
                simple_aggregations[output_name] = agg_func(keys[0])
            continue

        # e.g. agg(nw.mean('a')) # noqa: ERA001
        if (
            expr._depth != 1 or expr._root_names is None or expr._output_names is None
        ):  # pragma: no cover
            msg = "Safety assertion failed, please report a bug to https://github.com/narwhals-dev/narwhals/issues"
            raise AssertionError(msg)

        function_name = remove_prefix(expr._function_name, "col->")
        function_name = POLARS_TO_PYSPARK_AGGREGATIONS.get(function_name, function_name)

        for root_name, output_name in zip(expr._root_names, expr._output_names):
            agg_func = get_spark_function(function_name)
            simple_aggregations[output_name] = agg_func(root_name)
    agg_columns = [col_.alias(name) for name, col_ in simple_aggregations.items()]
    try:
        result_simple = grouped.agg(*agg_columns)
    except ValueError as exc:  # pragma: no cover
        msg = "Failed to aggregated - does your aggregation function return a scalar?"
        raise RuntimeError(msg) from exc
    return from_dataframe(result_simple)
