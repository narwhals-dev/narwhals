from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any, Protocol

from narwhals._compliant import LazyNamespace
from narwhals._compliant.typing import NativeExprT, NativeFrameT
from narwhals._sql.typing import SQLExprT, SQLLazyFrameT

if TYPE_CHECKING:
    from collections.abc import Iterable

    from narwhals.typing import PythonLiteral


class SQLNamespace(
    LazyNamespace[SQLLazyFrameT, SQLExprT, NativeFrameT],
    Protocol[SQLLazyFrameT, SQLExprT, NativeFrameT, NativeExprT],
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

    def when_then(self, *args: SQLExprT) -> SQLExprT:
        # Handle variable arguments for chaining:
        # - 2 args: (predicate, then) - single condition without otherwise
        # - 3 args: (predicate, then, otherwise) - single condition with otherwise
        # - 4+ args: (pred1, then1, pred2, then2, ..., [otherwise]) - chained conditions

        if len(args) == 2:
            # Simple case: when().then() without otherwise
            predicate, then = args

            def func(cols: list[NativeExprT]) -> NativeExprT:
                return self._when(cols[1], cols[0])

            return self._expr._from_elementwise_horizontal_op(func, then, predicate)

        if len(args) == 3:
            # Simple case: when().then().otherwise()
            predicate, then, otherwise = args

            def func_with_otherwise(cols: list[NativeExprT]) -> NativeExprT:
                return self._when(cols[1], cols[0], cols[2])

            return self._expr._from_elementwise_horizontal_op(
                func_with_otherwise, then, predicate, otherwise
            )

        # Chained conditions: (pred1, then1, pred2, then2, ..., [otherwise])
        def func_chained(cols: list[NativeExprT]) -> NativeExprT:
            from itertools import chain

            pairs = list(zip(cols[1::2], cols[::2]))
            reordered = list(chain.from_iterable(pairs))
            if len(cols) % 2 == 1:
                reordered.append(cols[-1])

            return self._when(reordered[0], reordered[1], *reordered[2:])

        from itertools import chain

        pairs = list(zip(args[1::2], args[::2]))
        call_args = list(chain.from_iterable(pairs))
        if len(args) % 2 == 1:
            call_args.append(args[-1])

        return self._expr._from_elementwise_horizontal_op(func_chained, *call_args)
