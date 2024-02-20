from __future__ import annotations

import collections
from typing import Any
from typing import Iterable

from narwhals.pandas_like.dataframe_object import LazyFrame
from narwhals.spec import DataFrame as DataFrameT
from narwhals.spec import GroupBy as GroupByT
from narwhals.spec import IntoExpr
from narwhals.spec import LazyFrame as LazyFrameT
from narwhals.spec import LazyGroupBy as LazyGroupByT
from narwhals.utils import dataframe_from_dict
from narwhals.utils import evaluate_simple_aggregation
from narwhals.utils import get_namespace
from narwhals.utils import horizontal_concat
from narwhals.utils import is_simple_aggregation
from narwhals.utils import parse_into_exprs


class GroupBy(GroupByT):
    def __init__(self, df: DataFrameT, keys: list[str], api_version: str) -> None:
        self._df = df
        self._keys = list(keys)
        self.api_version = api_version

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> DataFrameT:
        return (
            LazyGroupBy(self._df.lazy(), self._keys, self.api_version)
            .agg(*aggs, **named_aggs)
            .collect()
        )


class LazyGroupBy(LazyGroupByT):
    def __init__(self, df: LazyFrameT, keys: list[str], api_version: str) -> None:
        self._df = df
        self._keys = list(keys)
        self.api_version = api_version

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> LazyFrameT:
        df = self._df.dataframe  # type: ignore[attr-defined]
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
        implementation: str = self._df._implementation  # type: ignore[attr-defined]
        output_names: list[str] = self._keys
        for expr in exprs:
            expr_output_names = expr._output_names  # type: ignore[attr-defined]
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
                results_keys = expr.call(  # type: ignore[attr-defined]
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
            implementation=self._df._implementation,  # type: ignore[attr-defined]
        )
