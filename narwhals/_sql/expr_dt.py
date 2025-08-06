from __future__ import annotations

from typing import Generic

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import DateTimeNamespace
from narwhals._sql.typing import SQLExprT


class SQLExprDateTimeNamesSpace(
    LazyExprNamespace[SQLExprT], DateTimeNamespace[SQLExprT], Generic[SQLExprT]
): ...
