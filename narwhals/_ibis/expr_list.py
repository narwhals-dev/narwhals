from __future__ import annotations

from narwhals._compliant.any_namespace import ListNamespace
from narwhals._compliant.expr import LazyExprNamespace
from narwhals._ibis.expr import IbisExpr


class IbisExprListNamespace(LazyExprNamespace[IbisExpr], ListNamespace[IbisExpr]):
    def len(self) -> IbisExpr:
        return self.compliant._with_callable(lambda expr: expr.length())
