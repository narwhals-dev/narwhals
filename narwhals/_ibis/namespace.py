from __future__ import annotations

import operator
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence
from typing import cast

import ibis
import ibis.expr.types as ir

from narwhals._compliant import CompliantThen
from narwhals._compliant import LazyNamespace
from narwhals._compliant import LazyWhen
from narwhals._expression_parsing import combine_alias_output_names
from narwhals._expression_parsing import combine_evaluate_output_names
from narwhals._ibis.dataframe import IbisLazyFrame
from narwhals._ibis.expr import IbisExpr
from narwhals._ibis.selectors import IbisSelectorNamespace
from narwhals._ibis.utils import lit
from narwhals._ibis.utils import narwhals_to_native_dtype
from narwhals.utils import Implementation
from narwhals.utils import requires

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.typing import ConcatMethod
    from narwhals.utils import Version


class IbisNamespace(LazyNamespace[IbisLazyFrame, IbisExpr, "ir.Table"]):
    _implementation: Implementation = Implementation.IBIS

    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._version = version

    @property
    def selectors(self: Self) -> IbisSelectorNamespace:
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
        if how in {"horizontal", "diagonal"}:
            msg = f"{how!r} concat not supported for Ibis. Please join instead."
            raise NotImplementedError(msg)

        items = list(items)
        native_items = [item.native for item in items]
        first = items[0]
        schema = first.schema

        if not all(x.schema == schema for x in items[1:]):
            msg = "inputs should all have the same schema"
            raise TypeError(msg)
        return first._with_native(ibis.union(*native_items))

    @requires.backend_version((10, 0))
    def concat_str(
        self, *exprs: IbisExpr, separator: str, ignore_nulls: bool
    ) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Value]:
            cols = list(chain.from_iterable(expr(df) for expr in exprs))
            cols_casted = [s.cast("string") for s in cols]
            null_mask = [s.isnull() for s in cols]

            if not ignore_nulls:
                null_mask_result = reduce(operator.or_, null_mask)
                result = ibis.cases(
                    (
                        ~null_mask_result,
                        reduce(lambda x, y: x + separator + y, cols_casted),
                    ),
                    else_=None,
                )
            else:
                init_value, *values = [
                    cast("ir.StringValue", ibis.cases((~nm, col), else_=""))
                    for col, nm in zip(cols_casted, null_mask)
                ]

                separators = (
                    cast("ir.StringValue", ibis.cases((nm, ""), else_=separator))
                    for nm in null_mask[:-1]
                )
                result = reduce(
                    lambda x, y: x + y,
                    (s + v for s, v in zip(separators, values)),
                    init_value,
                )

            return [result]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def all_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Value]:
            cols = chain.from_iterable(expr(df) for expr in exprs)
            return [reduce(operator.and_, cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def any_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Value]:
            cols = chain.from_iterable(expr(df) for expr in exprs)
            return [reduce(operator.or_, cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def max_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Value]:
            cols = list(chain.from_iterable(expr(df) for expr in exprs))
            return [ibis.greatest(*cols).name(cols[0].get_name())]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def min_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Value]:
            cols = list(chain.from_iterable(expr(df) for expr in exprs))
            return [ibis.least(*cols).name(cols[0].get_name())]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def sum_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Value]:
            cols = [
                e.fill_null(lit(0)).name(e.get_name())
                for _expr in exprs
                for e in _expr(df)
            ]
            return [reduce(operator.add, cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def mean_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Value]:
            expr = (
                cast("ir.NumericColumn", e.fill_null(lit(0)))
                for _expr in exprs
                for e in _expr(df)
            )
            non_null = (
                cast("ir.NumericColumn", e.isnull().ifelse(lit(0), lit(1)))
                for _expr in exprs
                for e in _expr(df)
            )
            first_expr_name = exprs[0](df)[0].get_name()

            def _name_preserving_sum(
                e1: ir.NumericColumn, e2: ir.NumericColumn
            ) -> ir.NumericColumn:
                return cast("ir.NumericColumn", (e1 + e2).name(e1.get_name()))

            return [
                (
                    reduce(_name_preserving_sum, expr)
                    / reduce(_name_preserving_sum, non_null)
                ).name(first_expr_name)
            ]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    @requires.backend_version((10, 0))
    def when(self, predicate: IbisExpr) -> IbisWhen:
        return IbisWhen.from_expr(predicate, context=self)

    def lit(self: Self, value: Any, dtype: DType | type[DType] | None) -> IbisExpr:
        def func(_df: IbisLazyFrame) -> list[ir.Value]:
            if dtype is not None:
                ibis_dtype = narwhals_to_native_dtype(dtype, version=self._version)
                return [lit(value, ibis_dtype)]
            return [lit(value)]

        return self._expr(
            func,
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )

    def len(self: Self) -> IbisExpr:
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

    def __call__(self: Self, df: IbisLazyFrame) -> Sequence[ir.Value]:
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


class IbisThen(CompliantThen["IbisLazyFrame", "ir.Value", IbisExpr], IbisExpr): ...
