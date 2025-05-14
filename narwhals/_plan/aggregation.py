from __future__ import annotations

from narwhals._plan.common import ExprIR


class Agg(ExprIR): ...


class Count(Agg): ...


class First(Agg):
    """https://github.com/narwhals-dev/narwhals/issues/2526."""


class Last(Agg):
    """https://github.com/narwhals-dev/narwhals/issues/2526."""


class Max(Agg): ...


class Mean(Agg): ...


class Median(Agg): ...


class Min(Agg): ...


class NUnique(Agg): ...


class Quantile(Agg): ...


class Std(Agg): ...


class Sum(Agg): ...


class Var(Agg): ...


class OrderableAgg(Agg): ...


class ArgMin(OrderableAgg): ...


class ArgMax(OrderableAgg): ...
