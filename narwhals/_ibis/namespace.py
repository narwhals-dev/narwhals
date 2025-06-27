from __future__ import annotations

import operator
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Any, cast

import ibis
import ibis.expr.types as ir

from narwhals._compliant import LazyNamespace, LazyThen, LazyWhen
from narwhals._expression_parsing import (
    combine_alias_output_names,
    combine_evaluate_output_names,
)
from narwhals._ibis.dataframe import IbisLazyFrame
from narwhals._ibis.expr import IbisExpr
from narwhals._ibis.selectors import IbisSelectorNamespace
from narwhals._ibis.utils import lit, narwhals_to_native_dtype
from narwhals._utils import Implementation, requires

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from narwhals._utils import Version
    from narwhals.typing import ConcatMethod, IntoDType


class IbisNamespace(LazyNamespace[IbisLazyFrame, IbisExpr, "ir.Table"]):
    _implementation: Implementation = Implementation.IBIS

    def __init__(self, *, backend_version: tuple[int, ...], version: Version) -> None:
        self._backend_version = backend_version
        self._version = version

    @property
    def selectors(self) -> IbisSelectorNamespace:
        return IbisSelectorNamespace.from_namespace(self)

    @property
    def _expr(self) -> type[IbisExpr]:
        return IbisExpr

    @property
    def _lazyframe(self) -> type[IbisLazyFrame]:
        return IbisLazyFrame

    def concat(
        self, items: Iterable[IbisLazyFrame], *, how: ConcatMethod
    ) -> IbisLazyFrame:
        if how == "diagonal":
            msg = "diagonal concat not supported for Ibis. Please join instead."
            raise NotImplementedError(msg)

        items = list(items)
        native_items = [item.native for item in items]
        schema = items[0].schema
        if not all(x.schema == schema for x in items[1:]):
            msg = "inputs should all have the same schema"
            raise TypeError(msg)
        return self._lazyframe.from_native(ibis.union(*native_items), context=self)

    def concat_str(
        self, *exprs: IbisExpr, separator: str, ignore_nulls: bool
    ) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Value]:
            cols = list(chain.from_iterable(expr(df) for expr in exprs))
            cols_casted = [s.cast("string") for s in cols]

            if not ignore_nulls:
                result = cols_casted[0]
                for col in cols_casted[1:]:
                    result = result + separator + col
            else:
                sep = cast("ir.StringValue", lit(separator))
                result = sep.join(cols_casted)

            return [result]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def all_horizontal(self, *exprs: IbisExpr, ignore_nulls: bool) -> IbisExpr:
        def func(cols: Iterable[ir.Value]) -> ir.Value:
            it = (
                (col.fill_null(lit(True)) for col in cols)  # noqa: FBT003
                if ignore_nulls
                else cols
            )
            return reduce(operator.and_, it)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def any_horizontal(self, *exprs: IbisExpr, ignore_nulls: bool) -> IbisExpr:
        def func(cols: Iterable[ir.Value]) -> ir.Value:
            it = (
                (col.fill_null(lit(False)) for col in cols)  # noqa: FBT003
                if ignore_nulls
                else cols
            )
            return reduce(operator.or_, it)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def max_horizontal(self, *exprs: IbisExpr) -> IbisExpr:
        def func(cols: Iterable[ir.Value]) -> ir.Value:
            return ibis.greatest(*cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def min_horizontal(self, *exprs: IbisExpr) -> IbisExpr:
        def func(cols: Iterable[ir.Value]) -> ir.Value:
            return ibis.least(*cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def sum_horizontal(self, *exprs: IbisExpr) -> IbisExpr:
        def func(cols: Iterable[ir.Value]) -> ir.Value:
            cols = (col.fill_null(lit(0)) for col in cols)
            return reduce(operator.add, cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def mean_horizontal(self, *exprs: IbisExpr) -> IbisExpr:
        def func(cols: Iterable[ir.Value]) -> ir.Value:
            cols = list(cols)
            return reduce(operator.add, (col.fill_null(lit(0)) for col in cols)) / reduce(
                operator.add, (col.isnull().ifelse(lit(0), lit(1)) for col in cols)
            )

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    @requires.backend_version((10, 0))
    def when(self, predicate: IbisExpr) -> IbisWhen:
        return IbisWhen.from_expr(predicate, context=self)

    def lit(self, value: Any, dtype: IntoDType | None) -> IbisExpr:
        def func(_df: IbisLazyFrame) -> list[ir.Value]:
            ibis_dtype = narwhals_to_native_dtype(dtype, self._version) if dtype else None
            return [lit(value, ibis_dtype)]

        return self._expr(
            func,
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )

    def len(self) -> IbisExpr:
        def func(_df: IbisLazyFrame) -> list[ir.Value]:
            return [_df.native.count()]

        return self._expr(
            call=func,
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )


class IbisWhen(LazyWhen["IbisLazyFrame", "ir.Value", IbisExpr]):
    lit = lit

    @property
    def _then(self) -> type[IbisThen]:
        return IbisThen

    def __call__(self, df: IbisLazyFrame) -> Sequence[ir.Value]:
        is_expr = self._condition._is_expr
        condition = df._evaluate_expr(self._condition)
        then_ = self._then_value
        then = df._evaluate_expr(then_) if is_expr(then_) else lit(then_)
        other_ = self._otherwise_value
        if other_ is None:
            result = ibis.cases((condition, then))
        else:
            otherwise = df._evaluate_expr(other_) if is_expr(other_) else lit(other_)
            result = ibis.cases((condition, then), else_=otherwise)
        return [result]


class IbisThen(LazyThen["IbisLazyFrame", "ir.Value", IbisExpr], IbisExpr): ...
