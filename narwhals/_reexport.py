"""Re-export Narwhals functionality to avoid cyclical imports."""
from __future__ import annotations

from narwhals.dataframe import DataFrame, LazyFrame
from narwhals.expr import Expr
from narwhals.series import Series

__all__ = ["DataFrame", "Expr", "LazyFrame", "Series"]
