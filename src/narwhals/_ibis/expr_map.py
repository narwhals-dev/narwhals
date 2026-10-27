from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import MapNamespace
from narwhals._ibis.utils import lit

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from narwhals._ibis.expr import IbisExpr
    from narwhals.typing import NonNestedLiteral


class IbisExprMapNamespace(LazyExprNamespace["IbisExpr"], MapNamespace["IbisExpr"]):
    def get(self, key: NonNestedLiteral) -> IbisExpr:
        def func(expr: ir.MapColumn) -> ir.Value:
            return expr.get(lit(key))

        return self.compliant._with_callable(func)
