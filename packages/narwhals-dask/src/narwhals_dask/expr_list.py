from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
from narwhals.compliant import ListNamespace
from narwhals_dask._vendored import ExprNamespace, is_dtype_pyarrow, not_implemented

if TYPE_CHECKING:
    from collections.abc import Callable

    import dask.dataframe.dask_expr as dx
    import pandas as pd

    from narwhals.typing import NonNestedLiteral
    from narwhals_dask.expr import DaskExpr

    NwSeries = nw.Series[pd.Series]


def _apply_to_partition(
    partition: pd.Series, function: Callable[[NwSeries], NwSeries]
) -> pd.Series:
    series = nw.from_native(partition, series_only=True)
    return function(series).to_native()


class DaskExprListNamespace(ExprNamespace["DaskExpr"], ListNamespace["DaskExpr"]):
    def _map_partitions(self, function: Callable[[NwSeries], NwSeries]) -> DaskExpr:
        def call(native: dx.Series) -> dx.Series:
            if not is_dtype_pyarrow(native.dtype):
                msg = "Only pyarrow-backed lists are supported for Dask."
                raise NotImplementedError(msg)
            # An explicit `meta` skips Dask's inference, which would wrap errors (e.g.
            # `InvalidOperationError` for a mismatched item) in a `ValueError`.
            meta = _apply_to_partition(native._meta, function)
            return native.map_partitions(_apply_to_partition, function, meta=meta)

        return self.compliant._with_callable(call)

    def len(self) -> DaskExpr:
        return self._map_partitions(lambda series: series.list.len())

    def contains(self, item: NonNestedLiteral) -> DaskExpr:
        return self._map_partitions(lambda series: series.list.contains(item))

    def get(self, index: int) -> DaskExpr:
        return self._map_partitions(lambda series: series.list.get(index))

    def min(self) -> DaskExpr:
        return self._map_partitions(lambda series: series.list.min())

    def max(self) -> DaskExpr:
        return self._map_partitions(lambda series: series.list.max())

    def mean(self) -> DaskExpr:
        return self._map_partitions(lambda series: series.list.mean())

    def median(self) -> DaskExpr:
        return self._map_partitions(lambda series: series.list.median())

    def sum(self) -> DaskExpr:
        return self._map_partitions(lambda series: series.list.sum())

    def sort(self, *, descending: bool, nulls_last: bool) -> DaskExpr:
        return self._map_partitions(
            lambda series: series.list.sort(descending=descending, nulls_last=nulls_last)
        )

    unique = not_implemented()
