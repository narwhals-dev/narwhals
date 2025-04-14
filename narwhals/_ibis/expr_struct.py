from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from typing_extensions import Self

    from narwhals._ibis.expr import IbisExpr


class IbisExprStructNamespace:
    def __init__(self: Self, expr: IbisExpr) -> None:
        self._compliant_expr = expr

    def field(self: Self, name: str) -> IbisExpr:
        def func(_input: ir.Expr) -> ir.Expr:
            return _input[name]

        return self._compliant_expr._with_callable(func).alias(name)
