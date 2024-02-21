from __future__ import annotations

import collections
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable

from narwhals.pandas_like.dataframe import LazyFrame
from narwhals.pandas_like.utils import dataframe_from_dict
from narwhals.pandas_like.utils import evaluate_simple_aggregation
from narwhals.pandas_like.utils import get_namespace
from narwhals.pandas_like.utils import horizontal_concat
from narwhals.pandas_like.utils import is_simple_aggregation
from narwhals.pandas_like.utils import parse_into_exprs
from narwhals.spec import GroupBy as GroupByProtocol
from narwhals.spec import IntoExpr
from narwhals.spec import LazyGroupBy as LazyGroupByT

if TYPE_CHECKING:
    from narwhals.pandas_like.dataframe import DataFrame
    from narwhals.pandas_like.dataframe import LazyFrame


class GroupBy(GroupByProtocol):
    def __init__(self, df: DataFrame, keys: list[str], api_version: str) -> None:
        self._df = df
        self._keys = list(keys)
        self.api_version = api_version

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> DataFrame:
        return (
            LazyGroupBy(self._df.lazy(), self._keys, self.api_version)
            .agg(*aggs, **named_aggs)
            .collect()
        )


class LazyGroupBy(LazyGroupByT):
    def __init__(self, df: LazyFrame, keys: list[str], api_version: str) -> None:
        self._df = df
        self._keys = list(keys)
        self.api_version = api_version

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> LazyFrame:
        from narwhals.pandas_like.dataframe import LazyFrame

        df = self._df._dataframe
        exprs = parse_into_exprs(
            get_namespace(self._df),
            *aggs,
            **named_aggs,
        )
        grouped = df.groupby(
            list(self._keys),
            sort=False,
            as_index=False,
        )
        implementation: str = self._df._implementation
        output_names: list[str] = self._keys
        for expr in exprs:
            expr_output_names = expr._output_names
            if expr_output_names is None:
                msg = (
                    "Anonymous expressions are not supported in group_by.agg.\n"
                    "Instead of `pl.all()`, try using a named expression, such as "
                    "`pl.col('a', 'b')`\n"
                )
                raise ValueError(msg)
            output_names.extend(expr_output_names)

        dfs: list[Any] = []
        to_remove: list[int] = []
        for i, expr in enumerate(exprs):
            if is_simple_aggregation(expr):
                dfs.append(evaluate_simple_aggregation(expr, grouped))
                to_remove.append(i)
        exprs = [expr for i, expr in enumerate(exprs) if i not in to_remove]

        out: dict[str, list[Any]] = collections.defaultdict(list)
        for keys, df_keys in grouped:
            for key, name in zip(keys, self._keys):
                out[name].append(key)
            for expr in exprs:
                # TODO: it might be better to use groupby(...).apply
                # in this case, but I couldn't get the multi-output
                # case to work for cuDF.
                results_keys = expr.call(
                    LazyFrame(
                        df_keys,
                        api_version=self.api_version,
                        implementation=implementation,
                    )
                )
                for result_keys in results_keys:
                    out[result_keys.name].append(result_keys.item())

        results_keys = dataframe_from_dict(out, implementation=implementation)
        results_keys = horizontal_concat(
            [results_keys, *dfs], implementation=implementation
        ).loc[:, output_names]
        return LazyFrame(
            results_keys,
            api_version=self.api_version,
            implementation=self._df._implementation,
        )
