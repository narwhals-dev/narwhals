from __future__ import annotations

from narwhals._plan.common import ExprIR


class AggExpr(ExprIR): ...


class Count(AggExpr): ...


class First(AggExpr):
    """https://github.com/narwhals-dev/narwhals/issues/2526."""


class Last(AggExpr):
    """https://github.com/narwhals-dev/narwhals/issues/2526."""


class Max(AggExpr): ...


class Mean(AggExpr): ...


class Median(AggExpr): ...


class Min(AggExpr): ...


class NUnique(AggExpr): ...


class Quantile(AggExpr): ...


class Std(AggExpr): ...


class Sum(AggExpr): ...


class Var(AggExpr): ...
