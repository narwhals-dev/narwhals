from __future__ import annotations

from typing import Generic

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import DateTimeNamespace
from narwhals._sql.typing import SQLExprT


class SQLExprDateTimeNamesSpace(
    LazyExprNamespace[SQLExprT], DateTimeNamespace[SQLExprT], Generic[SQLExprT]
): 
    # TODO: @mp, since this is same as in SQLExprStringNamespace, should this be imported from that class? 
    def _function(self, name: str, *args: Any) -> SQLExprT:
        return self.compliant._function(name, *args) 
    
    def year(self) -> SQLExprT:
        return self.compliant._with_elementwise(lambda expr: self._function("year", expr))
