from __future__ import annotations

import collections
from typing import Any
from typing import Iterable

import pandas as pd

from polars_api_compat.pandas_like.dataframe_object import LazyFrame
from polars_api_compat.spec import DataFrame as DataFrameT
from polars_api_compat.spec import GroupBy as GroupByT
from polars_api_compat.spec import IntoExpr
from polars_api_compat.spec import LazyFrame as LazyFrameT
from polars_api_compat.spec import LazyGroupBy as LazyGroupByT
from polars_api_compat.utils import parse_into_exprs


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
        exprs = parse_into_exprs(
            self._df.__lazyframe_namespace__(),
            *aggs,
            **named_aggs,
        )
        grouped = self._df.dataframe.groupby(
            list(self._keys),
            sort=False,
            as_index=False,
        )

        # Do some fastpaths, if possible
        new_cols: list[pd.DataFrame] = []
        to_remove: list[int] = []
        for i, expr in enumerate(exprs):
            if (
                expr._function_name is not None  # type: ignore[attr-defined]
                and expr._depth is not None  # type: ignore[attr-defined]
                and expr._depth <= 2  # type: ignore[attr-defined]
                # todo: avoid this one?
                and expr._root_names is not None  # type: ignore[attr-defined]
            ):
                # We must have a simple aggregation, such as
                #     .agg(mean=pl.col('a').mean())
                # or
                #     .agg(pl.col('a').mean())
                if expr._root_names is None or expr.output_names is None:  # type: ignore[attr-defined]
                    msg = "Unreachable code, please report a bug"
                    raise AssertionError(msg)
                if len(expr._root_names) != len(expr.output_names):  # type: ignore[attr-defined]
                    msg = "Unreachable code, please report a bug"
                    raise AssertionError(msg)
                new_names = dict(zip(expr._root_names, expr.output_names))  # type: ignore[attr-defined]
                new_cols.append(
                    getattr(grouped[expr._root_names], expr._function_name)()[  # type: ignore[attr-defined]
                        expr._root_names  # type: ignore[attr-defined]
                    ].rename(columns=new_names),
                )
                to_remove.append(i)
        exprs = [expr for i, expr in enumerate(exprs) if i not in to_remove]

        out: dict[str, list[Any]] = collections.defaultdict(list)
        for key, _df in grouped:
            for _key, _name in zip(key, self._keys):
                out[_name].append(_key)
            for expr in exprs:
                result = expr.call(
                    LazyFrame(
                        _df,
                        api_version=self.api_version,
                        implementation=self._df._implementation,  # type: ignore[attr-defined]
                    )
                )
                for _result in result:
                    out[_result.name].append(_result.item())
        result = pd.DataFrame(out)
        result = pd.concat([result, *new_cols], axis=1, copy=False)
        return LazyFrame(
            result,
            api_version=self.api_version,
            implementation=self._df._implementation,  # type: ignore[attr-defined]
        )
