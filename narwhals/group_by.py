from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Iterator

from narwhals.utils import flatten
from narwhals.utils import tupleify

if TYPE_CHECKING:
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame
    from narwhals.typing import IntoExpr


class GroupBy:
    def __init__(self, df: DataFrame, *keys: str | Iterable[str]) -> None:
        self._df = df
        self._keys = flatten(keys)
        self._grouped = self._df._dataframe.group_by(self._keys)

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> DataFrame:
        """
        Compute aggregations for each group of a group by operation.

        Arguments:
            aggs: Aggregations to compute for each group of the group by operation,
                specified as positional arguments.

            named_aggs: Additional aggregations, specified as keyword arguments.

        Examples:
            Group by one column or by multiple columns and call `agg` to compute
            the grouped sum of another column.

            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame(
            ...     {
            ...         "a": ["a", "b", "a", "b", "c"],
            ...         "b": [1, 2, 1, 3, 3],
            ...         "c": [5, 4, 3, 2, 1],
            ...     }
            ... )
            >>> df_pl = pl.DataFrame(
            ...     {
            ...         "a": ["a", "b", "a", "b", "c"],
            ...         "b": [1, 2, 1, 3, 3],
            ...         "c": [5, 4, 3, 2, 1],
            ...     }
            ... )

            We define library agnostic functions:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.group_by("a").agg(nw.col("b").sum()).sort("a")
            ...     return nw.to_native(df)

            >>> def func_mult_col(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.group_by("a", "b").agg(nw.sum("c")).sort("a", "b")
            ...     return nw.to_native(df)

            We can then pass either pandas or Polars to `func` and `func_mult_col`:

            >>> func(df_pd)
               a  b
            0  a  2
            1  b  5
            2  c  3
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ str ┆ i64 │
            ╞═════╪═════╡
            │ a   ┆ 2   │
            │ b   ┆ 5   │
            │ c   ┆ 3   │
            └─────┴─────┘
            >>> func_mult_col(df_pd)
               a  b  c
            0  a  1  8
            1  b  2  4
            2  b  3  2
            3  c  3  1
            >>> func_mult_col(df_pl)
            shape: (4, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ str ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ a   ┆ 1   ┆ 8   │
            │ b   ┆ 2   ┆ 4   │
            │ b   ┆ 3   ┆ 2   │
            │ c   ┆ 3   ┆ 1   │
            └─────┴─────┴─────┘
        """
        aggs, named_aggs = self._df._flatten_and_extract(*aggs, **named_aggs)
        return self._df.__class__(
            self._grouped.agg(*aggs, **named_aggs),
        )

    def __iter__(self) -> Iterator[tuple[Any, DataFrame]]:
        import narwhals as nw

        yield from (
            (tupleify(key), nw.from_native(df, eager_only=True))
            for (key, df) in self._grouped.__iter__()
        )


class LazyGroupBy:
    def __init__(self, df: LazyFrame, *keys: str | Iterable[str]) -> None:
        self._df = df
        self._keys = keys
        self._grouped = self._df._dataframe.group_by(*self._keys)

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> LazyFrame:
        aggs, named_aggs = self._df._flatten_and_extract(*aggs, **named_aggs)
        return self._df.__class__(
            self._grouped.agg(*aggs, **named_aggs),
        )
