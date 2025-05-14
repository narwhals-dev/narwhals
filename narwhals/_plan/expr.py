from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Literal
import typing as t  # noqa: F401

from narwhals._plan.common import ExprIR


class Alias(ExprIR): ...


class Column(ExprIR): ...


class Literal(ExprIR): ...


class BinaryExpr(ExprIR):
    """Seems like the application of two exprs via an `Operator`."""


class Cast(ExprIR): ...


class Sort(ExprIR): ...


class SortBy(ExprIR):
    """https://github.com/narwhals-dev/narwhals/issues/2534."""


class Filter(ExprIR): ...


class Len(ExprIR): ...
