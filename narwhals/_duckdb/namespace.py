from __future__ import annotations

import operator
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import duckdb
from duckdb import CoalesceOperator, Expression, FunctionExpression
from duckdb.typing import BIGINT, VARCHAR

from narwhals._compliant import LazyNamespace, LazyThen, LazyWhen
from narwhals._duckdb.dataframe import DuckDBLazyFrame
from narwhals._duckdb.expr import DuckDBExpr
from narwhals._duckdb.selectors import DuckDBSelectorNamespace
from narwhals._duckdb.utils import concat_str, lit, narwhals_to_native_dtype, when
from narwhals._expression_parsing import (
    combine_alias_output_names,
    combine_evaluate_output_names,
)
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from narwhals._duckdb.expr import DuckDBWindowInputs
    from narwhals._utils import Version
    from narwhals.typing import ConcatMethod, IntoDType, NonNestedLiteral


class DuckDBNamespace(
    LazyNamespace[DuckDBLazyFrame, DuckDBExpr, duckdb.DuckDBPyRelation]
):
    _implementation: Implementation = Implementation.DUCKDB

    def __init__(self, *, backend_version: tuple[int, ...], version: Version) -> None:
        self._backend_version = backend_version
        self._version = version

    @property
    def selectors(self) -> DuckDBSelectorNamespace:
        return DuckDBSelectorNamespace.from_namespace(self)

    @property
    def _expr(self) -> type[DuckDBExpr]:
        return DuckDBExpr

    @property
    def _lazyframe(self) -> type[DuckDBLazyFrame]:
        return DuckDBLazyFrame

    def _with_elementwise(
        self, func: Callable[[Iterable[Expression]], Expression], *exprs: DuckDBExpr
    ) -> DuckDBExpr:
        def call(df: DuckDBLazyFrame) -> list[Expression]:
            cols = (col for _expr in exprs for col in _expr(df))
            return [func(cols)]

        def window_function(
            df: DuckDBLazyFrame, window_inputs: DuckDBWindowInputs
        ) -> list[Expression]:
            cols = (
                col for _expr in exprs for col in _expr.window_function(df, window_inputs)
            )
            return [func(cols)]

        return self._expr(
            call=call,
            window_function=window_function,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def concat(
        self, items: Iterable[DuckDBLazyFrame], *, how: ConcatMethod
    ) -> DuckDBLazyFrame:
        native_items = [item._native_frame for item in items]
        items = list(items)
        first = items[0]
        schema = first.schema
        if how == "vertical" and not all(x.schema == schema for x in items[1:]):
            msg = "inputs should all have the same schema"
            raise TypeError(msg)
        res = reduce(lambda x, y: x.union(y), native_items)
        return first._with_native(res)

    def concat_str(
        self, *exprs: DuckDBExpr, separator: str, ignore_nulls: bool
    ) -> DuckDBExpr:
        def func(df: DuckDBLazyFrame) -> list[Expression]:
            cols = list(chain.from_iterable(expr(df) for expr in exprs))
            if not ignore_nulls:
                null_mask_result = reduce(operator.or_, (s.isnull() for s in cols))
                cols_separated = [
                    y
                    for x in [
                        (col.cast(VARCHAR),)
                        if i == len(cols) - 1
                        else (col.cast(VARCHAR), lit(separator))
                        for i, col in enumerate(cols)
                    ]
                    for y in x
                ]
                return [when(~null_mask_result, concat_str(*cols_separated))]
            else:
                return [concat_str(*cols, separator=separator)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def all_horizontal(self, *exprs: DuckDBExpr) -> DuckDBExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            return reduce(operator.and_, cols)

        return self._with_elementwise(func, *exprs)

    def any_horizontal(self, *exprs: DuckDBExpr) -> DuckDBExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            return reduce(operator.or_, cols)

        return self._with_elementwise(func, *exprs)

    def max_horizontal(self, *exprs: DuckDBExpr) -> DuckDBExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            return FunctionExpression("greatest", *cols)

        return self._with_elementwise(func, *exprs)

    def min_horizontal(self, *exprs: DuckDBExpr) -> DuckDBExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            return FunctionExpression("least", *cols)

        return self._with_elementwise(func, *exprs)

    def sum_horizontal(self, *exprs: DuckDBExpr) -> DuckDBExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            return reduce(operator.add, (CoalesceOperator(col, lit(0)) for col in cols))

        return self._with_elementwise(func, *exprs)

    def mean_horizontal(self, *exprs: DuckDBExpr) -> DuckDBExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            cols = list(cols)
            return reduce(
                operator.add, (CoalesceOperator(col, lit(0)) for col in cols)
            ) / reduce(operator.add, (col.isnotnull().cast(BIGINT) for col in cols))

        return self._with_elementwise(func, *exprs)

    def when(self, predicate: DuckDBExpr) -> DuckDBWhen:
        return DuckDBWhen.from_expr(predicate, context=self)

    def lit(self, value: NonNestedLiteral, dtype: IntoDType | None) -> DuckDBExpr:
        def func(_df: DuckDBLazyFrame) -> list[Expression]:
            if dtype is not None:
                return [
                    lit(value).cast(
                        narwhals_to_native_dtype(dtype, version=self._version)  # type: ignore[arg-type]
                    )
                ]
            return [lit(value)]

        return self._expr(
            func,
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )

    def len(self) -> DuckDBExpr:
        def func(_df: DuckDBLazyFrame) -> list[Expression]:
            return [FunctionExpression("count")]

        return self._expr(
            call=func,
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )


class DuckDBWhen(LazyWhen["DuckDBLazyFrame", Expression, DuckDBExpr]):
    @property
    def _then(self) -> type[DuckDBThen]:
        return DuckDBThen

    def __call__(self, df: DuckDBLazyFrame) -> Sequence[Expression]:
        self.when = when
        self.lit = lit
        return super().__call__(df)

    def _window_function(
        self, df: DuckDBLazyFrame, window_inputs: DuckDBWindowInputs
    ) -> Sequence[Expression]:
        self.when = when
        self.lit = lit
        return super()._window_function(df, window_inputs)


class DuckDBThen(LazyThen["DuckDBLazyFrame", Expression, DuckDBExpr], DuckDBExpr): ...
