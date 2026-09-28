from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import ListNamespace
from narwhals._pandas_like.utils import is_dtype_pyarrow
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    import dask.dataframe.dask_expr as dx

    from narwhals._dask.dataframe import Incomplete
    from narwhals._dask.expr import DaskExpr
    from narwhals.typing import NonNestedLiteral


def _contains(native: pd.Series, item: NonNestedLiteral) -> pd.Series:
    from narwhals._arrow.utils import list_contains

    array: Incomplete = native.array
    result = pd.arrays.ArrowExtensionArray(list_contains(array._pa_array, item))
    return pd.Series(result, index=native.index, name=native.name)


class DaskExprListNamespace(LazyExprNamespace["DaskExpr"], ListNamespace["DaskExpr"]):
    def contains(self, item: NonNestedLiteral) -> DaskExpr:
        def func(expr: dx.Series) -> dx.Series:
            if not is_dtype_pyarrow(expr.dtype):  # pragma: no cover
                msg = "Only pyarrow-backed lists are supported for Dask."
                raise NotImplementedError(msg)
            return expr.map_partitions(_contains, item, meta=(expr.name, "bool[pyarrow]"))

        return self.compliant._with_callable(func)

    len = not_implemented()
    unique = not_implemented()
    get = not_implemented()
    min = not_implemented()
    max = not_implemented()
    mean = not_implemented()
    median = not_implemented()
    sum = not_implemented()
    sort = not_implemented()
