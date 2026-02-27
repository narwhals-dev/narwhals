from __future__ import annotations

from narwhals._plan.polars.dataframe import PolarsDataFrame as DataFrame
from narwhals._plan.polars.lazyframe import PolarsLazyFrame as LazyFrame
from narwhals._plan.polars.namespace import PolarsNamespace as Namespace

__all__ = ["DataFrame", "LazyFrame", "Namespace"]
