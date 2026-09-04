from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import DateTimeNamespace
from narwhals._sql.typing import SQLExprT

if TYPE_CHECKING:
    from narwhals._compliant.expr import NativeExpr


class SQLExprDateTimeNamesSpace(
    LazyExprNamespace[SQLExprT], DateTimeNamespace[SQLExprT], Generic[SQLExprT]
):
    def _function(self, name: str, *args: Any) -> NativeExpr:
        return self.compliant._function(name, *args)  # type: ignore[no-any-return]

    def year(self) -> SQLExprT:
        return self.compliant._with_elementwise(lambda expr: self._function("year", expr))  # pyrefly: ignore[bad-argument-type]  # pyrefly-issues/04-callable-typevar-through-nested-attr.md

    def month(self) -> SQLExprT:
        return self.compliant._with_elementwise(
            lambda expr: self._function("month", expr)  # pyrefly: ignore[bad-argument-type]  # pyrefly-issues/04-callable-typevar-through-nested-attr.md
        )

    def day(self) -> SQLExprT:
        return self.compliant._with_elementwise(lambda expr: self._function("day", expr))  # pyrefly: ignore[bad-argument-type]  # pyrefly-issues/04-callable-typevar-through-nested-attr.md

    def hour(self) -> SQLExprT:
        return self.compliant._with_elementwise(lambda expr: self._function("hour", expr))  # pyrefly: ignore[bad-argument-type]  # pyrefly-issues/04-callable-typevar-through-nested-attr.md

    def minute(self) -> SQLExprT:
        return self.compliant._with_elementwise(
            lambda expr: self._function("minute", expr)  # pyrefly: ignore[bad-argument-type]  # pyrefly-issues/04-callable-typevar-through-nested-attr.md
        )

    def second(self) -> SQLExprT:
        return self.compliant._with_elementwise(
            lambda expr: self._function("second", expr)  # pyrefly: ignore[bad-argument-type]  # pyrefly-issues/04-callable-typevar-through-nested-attr.md
        )

    def ordinal_day(self) -> SQLExprT:
        return self.compliant._with_elementwise(
            lambda expr: self._function("dayofyear", expr)  # pyrefly: ignore[bad-argument-type]  # pyrefly-issues/04-callable-typevar-through-nested-attr.md
        )

    def date(self) -> SQLExprT:
        return self.compliant._with_elementwise(
            lambda expr: self._function("to_date", expr)  # pyrefly: ignore[bad-argument-type]  # pyrefly-issues/04-callable-typevar-through-nested-attr.md
        )
