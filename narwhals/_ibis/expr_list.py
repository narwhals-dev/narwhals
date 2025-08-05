from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import ListNamespace

if TYPE_CHECKING:
    from narwhals._ibis.expr import IbisExpr


class IbisExprListNamespace(LazyExprNamespace["IbisExpr"], ListNamespace["IbisExpr"]):
    def len(self) -> IbisExpr:
        return self.compliant._with_callable(lambda expr: expr.length())

    def unique(self) -> IbisExpr:
        return self.compliant._with_callable(lambda expr: expr.unique())

    def contains(self, item: Any) -> IbisExpr:
        return self.compliant._with_callable(lambda expr: expr.contains(item))
