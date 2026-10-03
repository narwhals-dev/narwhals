from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import ListNamespace
from narwhals._pandas_like.series import PandasLikeSeries
from narwhals._pandas_like.utils import is_dtype_pyarrow
from narwhals._utils import Implementation, not_implemented

if TYPE_CHECKING:
    import dask.dataframe.dask_expr as dx
    import pandas as pd

    from narwhals._dask.expr import DaskExpr
    from narwhals._utils import Version
    from narwhals.typing import NonNestedLiteral


def _list_partition(
    partition: pd.Series, version: Version, method: str, *args: Any, **kwargs: Any
) -> pd.Series:
    series = PandasLikeSeries(
        partition, implementation=Implementation.PANDAS, version=version
    )
    return getattr(series.list, method)(*args, **kwargs).native


class DaskExprListNamespace(LazyExprNamespace["DaskExpr"], ListNamespace["DaskExpr"]):
    def _map_partitions(self, method: str, *args: Any, **kwargs: Any) -> DaskExpr:
        version = self.compliant._version

        def func(expr: dx.Series) -> dx.Series:
            if not is_dtype_pyarrow(expr.dtype):
                msg = "Only pyarrow-backed lists are supported for Dask."
                raise NotImplementedError(msg)
            # An explicit `meta` skips Dask's inference, which would wrap errors (e.g.
            # `InvalidOperationError` for a mismatched item) in a `ValueError`.
            meta = _list_partition(expr._meta, version, method, *args, **kwargs)
            return expr.map_partitions(
                _list_partition, version, method, *args, meta=meta, **kwargs
            )

        return self.compliant._with_callable(func)

    def len(self) -> DaskExpr:
        return self._map_partitions("len")

    def contains(self, item: NonNestedLiteral) -> DaskExpr:
        return self._map_partitions("contains", item)

    def get(self, index: int) -> DaskExpr:
        return self._map_partitions("get", index)

    def min(self) -> DaskExpr:
        return self._map_partitions("min")

    def max(self) -> DaskExpr:
        return self._map_partitions("max")

    def mean(self) -> DaskExpr:
        return self._map_partitions("mean")

    def median(self) -> DaskExpr:
        return self._map_partitions("median")

    def sum(self) -> DaskExpr:
        return self._map_partitions("sum")

    def sort(self, *, descending: bool, nulls_last: bool) -> DaskExpr:
        return self._map_partitions("sort", descending=descending, nulls_last=nulls_last)

    unique = not_implemented()
