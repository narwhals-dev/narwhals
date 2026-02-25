from __future__ import annotations

from narwhals._plan.polars.dataframe import PolarsDataFrame as DataFrame
from narwhals._plan.polars.lazyframe import PolarsLazyFrame as LazyFrame

__all__ = ["DataFrame", "LazyFrame"]
