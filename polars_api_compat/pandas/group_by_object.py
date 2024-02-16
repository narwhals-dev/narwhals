from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from polars_api_compat.pandas.dataframe_object import DataFrame
from polars_api_compat.spec import (
    DataFrame as DataFrameT,
    LazyFrame as LazyFrameT,
    GroupBy as GroupByT,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class GroupBy(GroupByT):
    def __init__(
        self, df: DataFrameT | LazyFrameT, keys: Sequence[str], api_version: str
    ) -> None:
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

    def _validate_booleanness(self) -> None:
        if not (
            (self._df.drop(columns=self._keys).dtypes == "bool")
            | (self._df.drop(columns=self._keys).dtypes == "boolean")
        ).all():
            msg = (
                "'function' can only be called on DataFrame where all dtypes are 'bool'"
            )
            raise TypeError(
                msg,
            )

    def _to_dataframe(self, result: pd.DataFrame) -> DataFrame:
        return DataFrame(
            result,
            api_version=self.api_version,
        )

    def size(self) -> DataFrame:
        return self._to_dataframe(self._grouped.size())

    def any(self) -> DataFrame:
        self._validate_booleanness()
        result = self._grouped.any()
        self._validate_result(result)
        return self._to_dataframe(result)

    def all(self) -> DataFrame:
        self._validate_booleanness()
        result = self._grouped.all()
        self._validate_result(result)
        return self._to_dataframe(result)

    def min(self) -> DataFrame:
        result = self._grouped.min()
        self._validate_result(result)
        return self._to_dataframe(result)

    def max(self) -> DataFrame:
        result = self._grouped.max()
        self._validate_result(result)
        return self._to_dataframe(result)

    def sum(self) -> DataFrame:
        result = self._grouped.sum()
        self._validate_result(result)
        return self._to_dataframe(result)

    def prod(self) -> DataFrame:
        result = self._grouped.prod()
        self._validate_result(result)
        return self._to_dataframe(result)

    def median(self) -> DataFrame:
        result = self._grouped.median()
        self._validate_result(result)
        return self._to_dataframe(result)

    def mean(self) -> DataFrame:
        result = self._grouped.mean()
        self._validate_result(result)
        return self._to_dataframe(result)

    def std(
        self,
        *,
        correction: float = 1.0,
    ) -> DataFrame:
        result = self._grouped.std()
        self._validate_result(result)
        return self._to_dataframe(result)

    def var(
        self,
        *,
        correction: float = 1.0,
    ) -> DataFrame:
        result = self._grouped.var()
        self._validate_result(result)
        return self._to_dataframe(result)

    def agg(
        self,
        *aggregations: Any,  # todo
    ) -> DataFrame:
        import collections

        aggs = []
        for aggregation in aggregations:
            if isinstance(aggregation, (list, tuple)):
                aggs.extend(aggregation)
            else:
                aggs.append(aggregation)

        out = collections.defaultdict(list)
        for key, _df in self._grouped:
            for _key, _name in zip(key, self._keys):
                out[_name].append(_key)
            for aggregation in aggs:
                result = aggregation.call(
                    DataFrame(
                        _df,
                        api_version=self.api_version,
                    )
                )
                for _result in result:
                    out[_result.name].append(_result.column.item())
        return self._to_dataframe(pd.DataFrame(out))
