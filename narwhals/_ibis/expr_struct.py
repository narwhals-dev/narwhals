from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant.any_namespace import StructNamespace
from narwhals._compliant.expr import LazyExprNamespace
from narwhals._ibis.expr import IbisExpr

if TYPE_CHECKING:
    import ibis.expr.types as ir


class IbisExprStructNamespace(LazyExprNamespace[IbisExpr], StructNamespace[IbisExpr]):
    def field(self, name: str) -> IbisExpr:
        def func(expr: ir.StructColumn) -> ir.Column:
            return expr[name]

        return self.compliant._with_callable(func).alias(name)
