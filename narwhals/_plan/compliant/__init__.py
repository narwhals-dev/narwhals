"""Second-cousin-to [`narwhals._compliant`].

`CompliantExpr` has the biggest facelift, where `ExprIR` ensures all `CompliantExpr`(s) have exactly one output.

All methods share a similar signature, with variation being the type used for `node`:

    class CompliantExpr(Protocol[FrameT]):
        def over(self, node: Over, frame: FrameT, name: str) -> Self: ...

By the time `CompliantExpr.over` is called, we will have already:
1. Recursively expanded any expressions nested within `node`
2. Validated all selected names exist in `frame`
3. Determine the **single output name** for the expression, passed here as `name`

[`narwhals._compliant`]: https://github.com/narwhals-dev/narwhals/blob/71c77b93181a420d4f8229d6ddaddc5eec215740/narwhals/_compliant/__init__.py
"""

from __future__ import annotations

from narwhals._plan.compliant import (
    accessors,
    broadcast,
    dataframe,
    expr,
    group_by,
    io,
    lazyframe,
    namespace,
    ranges,
    scalar,
    series,
)
from narwhals._plan.compliant.dataframe import CompliantDataFrame
from narwhals._plan.compliant.expr import CompliantExpr
from narwhals._plan.compliant.group_by import CompliantGroupBy
from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
from narwhals._plan.compliant.namespace import CompliantNamespace
from narwhals._plan.compliant.scalar import CompliantScalar
from narwhals._plan.compliant.series import CompliantSeries

__all__ = (
    "CompliantDataFrame",
    "CompliantExpr",
    "CompliantGroupBy",
    "CompliantLazyFrame",
    "CompliantNamespace",
    "CompliantScalar",
    "CompliantSeries",
    "accessors",
    "broadcast",
    "dataframe",
    "expr",
    "group_by",
    "io",
    "lazyframe",
    "namespace",
    "ranges",
    "scalar",
    "series",
)
