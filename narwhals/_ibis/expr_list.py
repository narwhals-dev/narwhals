from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narwhals._ibis.expr import IbisExpr


class IbisExprListNamespace:
    def __init__(self, expr: IbisExpr) -> None:
        self._compliant_expr = expr

    def len(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.length())
