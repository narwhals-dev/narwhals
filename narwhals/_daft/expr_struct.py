from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narwhals._daft.expr import DaftExpr


class DaftExprStructNamespace:
    def __init__(self, expr: DaftExpr) -> None:
        self._compliant_expr = expr

    def field(self, name: str) -> DaftExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.struct.get(name)
        ).alias(name)
