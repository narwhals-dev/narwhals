from __future__ import annotations

import collections
import functools
from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable

from narwhals.pandas_like.dataframe import LazyFrame
from narwhals.pandas_like.utils import dataframe_from_dict
from narwhals.pandas_like.utils import evaluate_simple_aggregation
from narwhals.pandas_like.utils import get_namespace
from narwhals.pandas_like.utils import horizontal_concat
from narwhals.pandas_like.utils import is_simple_aggregation
from narwhals.pandas_like.utils import item
from narwhals.pandas_like.utils import parse_into_exprs
from narwhals.spec import GroupBy as GroupByProtocol
from narwhals.spec import IntoExpr
from narwhals.spec import LazyGroupBy as LazyGroupByProtocol

if TYPE_CHECKING:
    from narwhals.pandas_like.dataframe import DataFrame
    from narwhals.pandas_like.dataframe import LazyFrame
    from narwhals.pandas_like.expr import Expr


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


class LazyGroupBy(LazyGroupByProtocol):
    def __init__(self, df: LazyFrame, keys: list[str], api_version: str) -> None:
        self._df = df
        self._keys = list(keys)
        self.api_version = api_version

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> LazyFrame:
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
        output_names: list[str] = copy(self._keys)
        for expr in exprs:
            if expr._output_names is None:
                msg = (
                    "Anonymous expressions are not supported in group_by.agg.\n"
                    "Instead of `pl.all()`, try using a named expression, such as "
                    "`pl.col('a', 'b')`\n"
                )
                raise ValueError(msg)
            output_names.extend(expr._output_names)

        dfs: list[Any] = []
        to_remove: list[int] = []
        for i, expr in enumerate(exprs):
            if is_simple_aggregation(expr):
                dfs.append(evaluate_simple_aggregation(expr, grouped))
                to_remove.append(i)
        exprs = [expr for i, expr in enumerate(exprs) if i not in to_remove]

        out: dict[str, list[Any]] = collections.defaultdict(list)
        for keys, _ in grouped:
            for key, name in zip(keys, self._keys):
                out[name].append(key)
        for expr in exprs:
            if len(expr._output_names) != 1:
                msg = (
                    "Multi-output non-elementary aggregations are not supported in group_by.agg.\n"
                    "Instead of `pl.col('a', 'b').sum()`, please separate them, e.g. "
                    "`.agg(pl.col('a').sum(), pl.col('b').sum())`\n"
                )
                raise ValueError(msg)

            def func(df: Any, expr: Expr) -> Any:
                return item(expr._call(self._from_dataframe(df))[0]._series)

            res = grouped.apply(functools.partial(func, expr=expr))
            for key in self._keys:
                if key in res.columns:
                    res = res.drop(columns=[key])
            res = res.set_axis(expr._output_names, axis=1)
            dfs.append(res)

        results_keys = dataframe_from_dict(out, implementation=implementation)
        results_keys = horizontal_concat(
            [results_keys, *dfs], implementation=implementation
        ).loc[:, output_names]
        return self._from_dataframe(results_keys)

    def _from_dataframe(self, df: DataFrame) -> LazyFrame:
        from narwhals.pandas_like.dataframe import LazyFrame

        return LazyFrame(
            df, api_version=self.api_version, implementation=self._df._implementation
        )
