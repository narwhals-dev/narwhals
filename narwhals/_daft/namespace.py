from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import daft

from narwhals._compliant import CompliantThen, LazyWhen
from narwhals._compliant.namespace import LazyNamespace
from narwhals._daft.dataframe import DaftLazyFrame
from narwhals._daft.expr import DaftExpr
from narwhals._daft.selectors import DaftSelectorNamespace
from narwhals._daft.utils import lit, narwhals_to_native_dtype
from narwhals._expression_parsing import (
    combine_alias_output_names,
    combine_evaluate_output_names,
)
from narwhals.utils import Implementation, not_implemented

if TYPE_CHECKING:
    from narwhals.dtypes import DType
    from narwhals.typing import ConcatMethod
    from narwhals.utils import Version


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
        def func(_df: DaftLazyFrame) -> list[daft.Expression]:
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

    def all_horizontal(self, *exprs: DaftExpr) -> DaftExpr:
        def func(df: DaftLazyFrame) -> list[daft.Expression]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [reduce(operator.and_, cols)]

        return DaftExpr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def any_horizontal(self, *exprs: DaftExpr) -> DaftExpr:
        def func(df: DaftLazyFrame) -> list[daft.Expression]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [reduce(operator.or_, cols)]

        return DaftExpr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    max_horizontal = not_implemented()
    min_horizontal = not_implemented()

    def sum_horizontal(self, *exprs: DaftExpr) -> DaftExpr:
        def func(df: DaftLazyFrame) -> list[daft.Expression]:
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
        def func(df: DaftLazyFrame) -> list[daft.Expression]:
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
        def func(_df: DaftLazyFrame) -> list[daft.Expression]:
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


class DaftWhen(LazyWhen[DaftLazyFrame, daft.Expression, DaftExpr]):
    @property
    def _then(self) -> type[DaftThen]:
        return DaftThen

    def __call__(self, df: DaftLazyFrame) -> Sequence[daft.Expression]:
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


class DaftThen(CompliantThen[DaftLazyFrame, daft.Expression, DaftExpr], DaftExpr): ...
