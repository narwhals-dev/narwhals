from __future__ import annotations

from typing import Any, Generic

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import StringNamespace
from narwhals._sql.typing import SQLExprT


class SQLExprStringNamespace(
    LazyExprNamespace["SQLExprT"], StringNamespace["SQLExprT"], Generic[SQLExprT]
):
    def _lit(self, value: Any) -> SQLExprT:
        return self.compliant._lit(value)

    def _function(self, name: str, *args: Any) -> SQLExprT:
        return self.compliant._function(name, *args)

    def to_lowercase(self) -> SQLExprT:
        return self.compliant._with_elementwise(
            lambda expr: self._function("lower", expr)
        )

    def to_uppercase(self) -> SQLExprT:
        return self.compliant._with_elementwise(
            lambda expr: self._function("upper", expr)
        )

    def starts_with(self, prefix: str) -> SQLExprT:
        return self.compliant._with_elementwise(
            lambda expr: self._function("starts_with", expr, self._lit(prefix))
        )

    def ends_with(self, suffix: str) -> SQLExprT:
        return self.compliant._with_elementwise(
            lambda expr: self._function("ends_with", expr, self._lit(suffix))
        )
