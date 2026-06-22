from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.expressions.name import IRNameNamespace
from narwhals._plan.expressions.namespace import ExprNamespace

if TYPE_CHECKING:
    from narwhals._compliant.typing import AliasName
    from narwhals._plan.expr import Expr


class ExprNameNamespace(ExprNamespace[IRNameNamespace]):
    @property
    def _ir_namespace(self) -> type[IRNameNamespace]:
        return IRNameNamespace

    def keep(self) -> Expr:
        return self._to_narwhals(self._ir.keep())

    def map(self, function: AliasName) -> Expr:
        """Define an alias by mapping a function over the original root column name."""
        return self._to_narwhals(self._ir.map(function))

    def prefix(self, prefix: str) -> Expr:
        """Add a prefix to the root column name."""
        return self._to_narwhals(self._ir.prefix(prefix))

    def suffix(self, suffix: str) -> Expr:
        """Add a suffix to the root column name."""
        return self._to_narwhals(self._ir.suffix(suffix))

    def to_lowercase(self) -> Expr:
        """Update the root column name to use lowercase characters."""
        return self._to_narwhals(self._ir.to_lowercase())

    def to_uppercase(self) -> Expr:
        """Update the root column name to use uppercase characters."""
        return self._to_narwhals(self._ir.to_uppercase())
