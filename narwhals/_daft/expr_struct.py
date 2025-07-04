from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narwhals._daft.expr import DaftExpr


class DaftExprStructNamespace:
    def __init__(self, expr: DaftExpr) -> None:
        self.compliant = expr

    def field(self, name: str) -> DaftExpr:
        return self.compliant._with_callable(
            lambda _input: _input.struct.get(name)
        ).alias(name)
