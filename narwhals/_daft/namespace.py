from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

import daft
from daft import Expression

from narwhals._compliant import LazyThen, LazyWhen
from narwhals._compliant.namespace import LazyNamespace
from narwhals._daft.dataframe import DaftLazyFrame
from narwhals._daft.expr import DaftExpr
from narwhals._daft.selectors import DaftSelectorNamespace
from narwhals._daft.utils import lit, narwhals_to_native_dtype
from narwhals._expression_parsing import (
    combine_alias_output_names,
    combine_evaluate_output_names,
)
from narwhals._utils import Implementation, not_implemented

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from narwhals._daft.expr import DaftWindowInputs
    from narwhals._utils import Version
    from narwhals.dtypes import DType
    from narwhals.typing import ConcatMethod


class DaftNamespace(LazyNamespace[DaftLazyFrame, DaftExpr, daft.DataFrame]):
    _implementation: Implementation = Implementation.DAFT

    def __init__(self, *, backend_version: tuple[int, ...], version: Version) -> None:
        self._backend_version = backend_version
        self._version = version

    @property
    def selectors(self) -> DaftSelectorNamespace:
        return DaftSelectorNamespace.from_namespace(self)

    @property
    def _expr(self) -> type[DaftExpr]:
        return DaftExpr

    @property
    def _lazyframe(self) -> type[DaftLazyFrame]:
        return DaftLazyFrame

    def lit(self, value: Any, dtype: DType | type[DType] | None) -> DaftExpr:
        def func(_df: DaftLazyFrame) -> list[Expression]:
            if dtype is not None:
                return [
                    lit(value).cast(
                        narwhals_to_native_dtype(
                            dtype, self._version, self._backend_version
                        )
                    )
                ]
            return [lit(value)]

        return DaftExpr(
            func,
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )

    def concat(
        self, items: Iterable[DaftLazyFrame], *, how: ConcatMethod
    ) -> DaftLazyFrame:
        list_items = list(items)
        native_items = (item._native_frame for item in items)
        if how == "diagonal":
            return DaftLazyFrame(
                reduce(lambda x, y: x.union_all_by_name(y), native_items),
                backend_version=self._backend_version,
                version=self._version,
            )
        first = list_items[0]
        schema = first.schema
        if how == "vertical" and not all(x.schema == schema for x in list_items[1:]):
            msg = "inputs should all have the same schema"
            raise TypeError(msg)
        res = reduce(lambda x, y: x.union(y), native_items)
        return first._with_native(res)

    concat_str = not_implemented()

    def all_horizontal(self, *exprs: DaftExpr, ignore_nulls: bool) -> DaftExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            it = (
                (daft.coalesce(col, lit(True)) for col in cols)  # noqa: FBT003
                if ignore_nulls
                else cols
            )
            return reduce(operator.and_, it)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def any_horizontal(self, *exprs: DaftExpr, ignore_nulls: bool) -> DaftExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            it = (
                (daft.coalesce(col, lit(False)) for col in cols)  # noqa: FBT003
                if ignore_nulls
                else cols
            )
            return reduce(operator.or_, it)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    max_horizontal = not_implemented()
    min_horizontal = not_implemented()

    def sum_horizontal(self, *exprs: DaftExpr) -> DaftExpr:
        def func(df: DaftLazyFrame) -> list[Expression]:
            cols = (col.fill_null(lit(0)) for _expr in exprs for col in _expr(df))
            return [reduce(operator.add, cols)]

        return DaftExpr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def mean_horizontal(self, *exprs: DaftExpr) -> DaftExpr:
        def func(df: DaftLazyFrame) -> list[Expression]:
            cols = [c for _expr in exprs for c in _expr(df)]
            return [
                (
                    reduce(operator.add, (col.fill_null(lit(0)) for col in cols))
                    / reduce(operator.add, ((~col.is_null()).cast("int") for col in cols))
                )
            ]

        return DaftExpr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def when(self, predicate: DaftExpr) -> DaftWhen:
        return DaftWhen.from_expr(predicate, context=self)

    def len(self) -> DaftExpr:
        def func(_df: DaftLazyFrame) -> list[Expression]:
            if not _df.columns:
                msg = "Cannot use `nw.len()` on Daft DataFrame with zero columns"
                raise ValueError(msg)
            return [daft.col(_df.columns[0]).count(mode="all")]

        return DaftExpr(
            call=func,
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )


class DaftWhen(LazyWhen[DaftLazyFrame, Expression, DaftExpr]):
    @property
    def _then(self) -> type[DaftThen]:
        return DaftThen

    def __call__(self, df: DaftLazyFrame) -> Sequence[Expression]:
        is_expr = self._condition._is_expr
        condition = df._evaluate_expr(self._condition)
        then_ = self._then_value
        then = df._evaluate_expr(then_) if is_expr(then_) else lit(then_)
        other_ = self._otherwise_value
        if other_ is None:
            result = condition.if_else(then, None)
        else:
            otherwise = df._evaluate_expr(other_) if is_expr(other_) else lit(other_)
            result = condition.if_else(then, otherwise)
        return [result]

    def _window_function(
        self, df: DaftLazyFrame, window_inputs: DaftWindowInputs
    ) -> Sequence[Expression]:
        is_expr = self._condition._is_expr
        condition = df._evaluate_window_expr(self._condition, window_inputs)
        then_ = self._then_value
        then = (
            df._evaluate_window_expr(then_, window_inputs)
            if is_expr(then_)
            else lit(then_)
        )
        other_ = self._otherwise_value
        if other_ is None:
            result = condition.if_else(then, None)
        else:
            otherwise = (
                df._evaluate_window_expr(other_, window_inputs)
                if is_expr(other_)
                else lit(other_)
            )
            result = condition.if_else(then, otherwise)
        return [result]


class DaftThen(LazyThen[DaftLazyFrame, Expression, DaftExpr], DaftExpr): ...
