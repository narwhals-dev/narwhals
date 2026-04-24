from __future__ import annotations

from narwhals._plan.polars import v1
from narwhals._plan.polars.dataframe import PolarsDataFrame as DataFrame
from narwhals._plan.polars.expr import PolarsExpr as Expr
from narwhals._plan.polars.lazyframe import (
    PolarsEvaluator as PlanEvaluator,
    PolarsLazyFrame as LazyFrame,
)
from narwhals._plan.polars.namespace import PolarsNamespace as Namespace
from narwhals._plan.polars.series import PolarsSeries as Series

Scalar = Expr
PlanResolver = None

__all__ = [
    "DataFrame",
    "Expr",
    "LazyFrame",
    "Namespace",
    "PlanEvaluator",
    "PlanResolver",
    "Scalar",
    "Series",
    "v1",
]
