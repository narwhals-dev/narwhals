from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import ListNamespace
from narwhals._pandas_like.series import PandasLikeSeries
from narwhals._pandas_like.utils import is_dtype_pyarrow
from narwhals._utils import Implementation, not_implemented

if TYPE_CHECKING:
    from collections.abc import Callable

    import dask.dataframe.dask_expr as dx
    import pandas as pd

    from narwhals._dask.expr import DaskExpr
    from narwhals._utils import Version
    from narwhals.typing import NonNestedLiteral


def _apply_to_partition(
    partition: pd.Series,
    version: Version,
    function: Callable[[PandasLikeSeries], PandasLikeSeries],
) -> pd.Series:
    series = PandasLikeSeries(
        partition, implementation=Implementation.PANDAS, version=version
    )
    return function(series).native


class DaskExprListNamespace(LazyExprNamespace["DaskExpr"], ListNamespace["DaskExpr"]):
    def _map_partitions(
        self, function: Callable[[PandasLikeSeries], PandasLikeSeries]
    ) -> DaskExpr:
        version = self.compliant._version

        def call(native: dx.Series) -> dx.Series:
            if not is_dtype_pyarrow(native.dtype):
                msg = "Only pyarrow-backed lists are supported for Dask."
                raise NotImplementedError(msg)
            # An explicit `meta` skips Dask's inference, which would wrap errors (e.g.
            # `InvalidOperationError` for a mismatched item) in a `ValueError`.
            meta = _apply_to_partition(native._meta, version, function)
            return native.map_partitions(
                _apply_to_partition, version, function, meta=meta
            )

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
