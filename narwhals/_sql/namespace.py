from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

from narwhals._compliant import LazyNamespace

if TYPE_CHECKING:
    from collections.abc import Iterable

    from narwhals._utils import Version


from typing import TYPE_CHECKING, Any

from narwhals._compliant.typing import NativeExprT, NativeFrameT
from narwhals._sql.expr import SQLExpr
from narwhals._sql.typing import SQLLazyFrameT

if TYPE_CHECKING:
    from collections.abc import Iterable

    from narwhals.typing import PythonLiteral


class SQLNamespace(
    LazyNamespace[SQLLazyFrameT, SQLExpr[SQLLazyFrameT, NativeExprT], NativeFrameT]
):
    def __init__(self, *, version: Version) -> None:
        self._version = version

    @property
    def _expr(self) -> type[SQLExpr[SQLLazyFrameT, NativeExprT]]: ...

    @property
    def _lazyframe(self) -> type[SQLLazyFrameT]: ...

    def _function(self, name: str, *args: NativeExprT | PythonLiteral) -> NativeExprT: ...
    def _lit(self, value: Any) -> NativeExprT: ...
    def _coalesce(self, *exprs: NativeExprT) -> NativeExprT: ...

    # Horizontal functions
    def any_horizontal(
        self, *exprs: SQLExpr[SQLLazyFrameT, NativeExprT], ignore_nulls: bool
    ) -> SQLExpr[SQLLazyFrameT, NativeExprT]:
        def func(cols: Iterable[NativeExprT]) -> NativeExprT:
            it = (
                (self._coalesce(col, self._lit(False)) for col in cols)  # noqa: FBT003
                if ignore_nulls
                else cols
            )
            return reduce(operator.or_, it)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def all_horizontal(
        self, *exprs: SQLExpr[SQLLazyFrameT, NativeExprT], ignore_nulls: bool
    ) -> SQLExpr[SQLLazyFrameT, NativeExprT]:
        def func(cols: Iterable[NativeExprT]) -> NativeExprT:
            it = (
                (self._coalesce(col, self._lit(True)) for col in cols)  # noqa: FBT003
                if ignore_nulls
                else cols
            )
            return reduce(operator.and_, it)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def max_horizontal(
        self, *exprs: SQLExpr[SQLLazyFrameT, NativeExprT]
    ) -> SQLExpr[SQLLazyFrameT, NativeExprT]:
        def func(cols: Iterable[NativeExprT]) -> NativeExprT:
            return self._function("greatest", *cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def min_horizontal(
        self, *exprs: SQLExpr[SQLLazyFrameT, NativeExprT]
    ) -> SQLExpr[SQLLazyFrameT, NativeExprT]:
        def func(cols: Iterable[NativeExprT]) -> NativeExprT:
            return self._function("least", *cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def sum_horizontal(
        self, *exprs: SQLExpr[SQLLazyFrameT, NativeExprT]
    ) -> SQLExpr[SQLLazyFrameT, NativeExprT]:
        def func(cols: Iterable[NativeExprT]) -> NativeExprT:
            return reduce(
                operator.add, (self._coalesce(col, self._lit(0)) for col in cols)
            )

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    # Other
    def coalesce(
        self, *exprs: SQLExpr[SQLLazyFrameT, NativeExprT]
    ) -> SQLExpr[SQLLazyFrameT, NativeExprT]:
        def func(cols: Iterable[NativeExprT]) -> NativeExprT:
            return self._coalesce(*cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)
