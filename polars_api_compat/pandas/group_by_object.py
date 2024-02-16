from __future__ import annotations
import collections

from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd
from polars_api_compat.utils import parse_into_exprs

from polars_api_compat.pandas.dataframe_object import DataFrame, LazyFrame
from polars_api_compat.spec import (
    DataFrame as DataFrameT,
    LazyFrame as LazyFrameT,
    GroupBy as GroupByT,
    LazyGroupBy as LazyGroupByT,
    IntoExpr,
)

if TYPE_CHECKING:
    pass


class GroupBy(GroupByT):
    def __init__(self, df: DataFrameT, keys: list[str], api_version: str) -> None:
        self._df = df.dataframe
        self._grouped = self._df.groupby(list(keys), sort=False, as_index=False)
        self._keys = list(keys)
        self.api_version = api_version

    def _validate_result(self, result: pd.DataFrame) -> None:
        failed_columns = self._df.columns.difference(result.columns)
        if len(failed_columns) > 0:  # pragma: no cover
            msg = "Groupby operation could not be performed on columns "
            f"{failed_columns}. Please drop them before calling group_by."
            raise AssertionError(
                msg,
            )

    def _to_dataframe(self, result: pd.DataFrame) -> DataFrame:
        return DataFrame(
            result,
            api_version=self.api_version,
        )

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> DataFrame:
        # todo: dedupe logic with LazyGroupBy.agg
        exprs = parse_into_exprs(
            self._df.__dataframe_namespace__(), *aggs, **named_aggs
        )
        out: dict[str, list[Any]] = collections.defaultdict(list)
        for key, _df in self._grouped:
            for _key, _name in zip(key, self._keys):
                out[_name].append(_key)
            for expr in exprs:
                result = expr.call(LazyFrame(_df, api_version=self.api_version))
                for _result in result:
                    out[_result.name].append(_result.series.item())
        return self._to_dataframe(pd.DataFrame(out))


class LazyGroupBy(LazyGroupByT):
    def __init__(self, df: LazyFrameT, keys: list[str], api_version: str) -> None:
        self._df = df
        self._grouped = self._df.dataframe.groupby(
            list(keys), sort=False, as_index=False
        )
        self._keys = list(keys)
        self.api_version = api_version

    def _validate_result(self, result: pd.DataFrame) -> None:
        failed_columns = self._df.dataframe.columns.difference(result.columns)
        if len(failed_columns) > 0:  # pragma: no cover
            msg = "Groupby operation could not be performed on columns "
            f"{failed_columns}. Please drop them before calling group_by."
            raise AssertionError(
                msg,
            )

    def _to_dataframe(self, result: pd.DataFrame) -> LazyFrameT:
        return LazyFrame(
            result,
            api_version=self.api_version,
        )

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> LazyFrameT:
        exprs = parse_into_exprs(
            self._df.__lazyframe_namespace__(), *aggs, **named_aggs
        )
        out: dict[str, list[Any]] = collections.defaultdict(list)
        for key, _df in self._grouped:
            for _key, _name in zip(key, self._keys):
                out[_name].append(_key)
            for expr in exprs:
                result = expr.call(LazyFrame(_df, api_version=self.api_version))
                for _result in result:
                    out[_result.name].append(_result.series.item())
        return self._to_dataframe(pd.DataFrame(out))
