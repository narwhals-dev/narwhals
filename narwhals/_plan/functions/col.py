from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan import expressions as ir, selectors as cs

if TYPE_CHECKING:
    from collections.abc import Iterable

    from narwhals._plan.expr import Expr

__all__ = ("col",)


def col(name: str | Iterable[str], *more_names: str) -> Expr:
    more = more_names
    if not more:
        if isinstance(name, str) or (len(more := tuple(name)) == 1 and (name := more[0])):
            return ir.col(name).to_narwhals()
        return cs.by_name(*more).as_expr()
    return cs.by_name(name, *more).as_expr()
