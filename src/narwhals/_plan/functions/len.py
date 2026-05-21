from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan import expressions as ir

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr


def len() -> Expr:
    return ir.Len().to_narwhals()
