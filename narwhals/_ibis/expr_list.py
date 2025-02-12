from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._ibis.expr import IbisExpr


class IbisExprListNamespace:
    def __init__(self: Self, expr: IbisExpr) -> None:
        self._compliant_expr = expr

    def len(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.length(),
            "len",
            expr_kind=self._compliant_expr._expr_kind,
        )
