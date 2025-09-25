from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any, Protocol, cast

from narwhals._compliant import LazyNamespace
from narwhals._compliant.typing import NativeExprT, NativeFrameT_co
from narwhals._expression_parsing import is_expr
from narwhals._sql.typing import SQLExprT, SQLLazyFrameT
from narwhals.functions import lit

if TYPE_CHECKING:
    from collections.abc import Iterable

    from narwhals.expr import Expr
    from narwhals.typing import NonNestedLiteral, PythonLiteral


class SQLNamespace(
    LazyNamespace[SQLLazyFrameT, SQLExprT, NativeFrameT_co],
    Protocol[SQLLazyFrameT, SQLExprT, NativeFrameT_co, NativeExprT],
):
    def _function(self, name: str, *args: NativeExprT | PythonLiteral) -> NativeExprT: ...
    def _lit(self, value: Any) -> NativeExprT: ...
    def _when(
        self,
        condition: NativeExprT,
        value: NativeExprT,
        otherwise: NativeExprT | None = None,
    ) -> NativeExprT: ...
    def _coalesce(self, *exprs: NativeExprT) -> NativeExprT: ...

    def evaluate_expr(self, data: Expr | NonNestedLiteral | Any, /) -> SQLExprT:
        if is_expr(data):
            expr = data(self)
            assert isinstance(expr, self._expr)  # noqa: S101
            return expr
        return cast("SQLExprT", lit(data)(self))

    # Horizontal functions
    def any_horizontal(self, *exprs: SQLExprT, ignore_nulls: bool) -> SQLExprT:
        def func(cols: Iterable[NativeExprT]) -> NativeExprT:
            if ignore_nulls:
                cols = (self._coalesce(col, self._lit(False)) for col in cols)
            return reduce(operator.or_, cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def all_horizontal(self, *exprs: SQLExprT, ignore_nulls: bool) -> SQLExprT:
        def func(cols: Iterable[NativeExprT]) -> NativeExprT:
            if ignore_nulls:
                cols = (self._coalesce(col, self._lit(True)) for col in cols)
            return reduce(operator.and_, cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def max_horizontal(self, *exprs: SQLExprT) -> SQLExprT:
        def func(cols: Iterable[NativeExprT]) -> NativeExprT:
            return self._function("greatest", *cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def min_horizontal(self, *exprs: SQLExprT) -> SQLExprT:
        def func(cols: Iterable[NativeExprT]) -> NativeExprT:
            return self._function("least", *cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def sum_horizontal(self, *exprs: SQLExprT) -> SQLExprT:
        def func(cols: Iterable[NativeExprT]) -> NativeExprT:
            return reduce(
                operator.add, (self._coalesce(col, self._lit(0)) for col in cols)
            )

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    # Other
    def coalesce(self, *exprs: SQLExprT) -> SQLExprT:
        def func(cols: Iterable[NativeExprT]) -> NativeExprT:
            return self._coalesce(*cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def when_then(
        self, predicate: SQLExprT, then: SQLExprT, otherwise: SQLExprT | None = None
    ) -> SQLExprT:
        def func(cols: list[NativeExprT]) -> NativeExprT:
            return self._when(cols[1], cols[0])

        def func_with_otherwise(cols: list[NativeExprT]) -> NativeExprT:
            return self._when(cols[1], cols[0], cols[2])

        if otherwise is None:
            return self._expr._from_elementwise_horizontal_op(func, then, predicate)
        return self._expr._from_elementwise_horizontal_op(
            func_with_otherwise, then, predicate, otherwise
        )
