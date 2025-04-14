from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Sequence

from narwhals._compliant import CompliantThen
from narwhals._compliant import LazyNamespace
from narwhals._compliant import LazyWhen
from narwhals._expression_parsing import combine_alias_output_names
from narwhals._expression_parsing import combine_evaluate_output_names
from narwhals._ibis.dataframe import IbisLazyFrame
from narwhals._ibis.expr import IbisExpr
from narwhals._ibis.selectors import IbisSelectorNamespace
from narwhals._ibis.utils import narwhals_to_native_dtype
from narwhals.dependencies import get_ibis
from narwhals.utils import Implementation

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from typing_extensions import Self

    from narwhals.dtypes import DType
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
        self: Self,
        items: Iterable[IbisLazyFrame],
        *,
        how: Literal["horizontal", "vertical", "diagonal"],
    ) -> IbisLazyFrame:
        ibis = get_ibis()

        if how == "horizontal":
            msg = "horizontal concat not supported for Ibis. Please join instead"
            raise TypeError(msg)
        if how == "diagonal":
            msg = "diagonal concat not supported for Ibis. Please join instead"
            raise NotImplementedError(msg)

        native_items = [item.native for item in items]
        items = list(items)
        first = items[0]
        schema = first.schema

        if how == "vertical" and not all(x.schema == schema for x in items[1:]):
            msg = "inputs should all have the same schema"
            raise TypeError(msg)

        res = ibis.union(native_items[0], *native_items[1:])
        return first._with_native(res)

    def concat_str(  # TODO(rwhitten577): IMPLEMENT
        self: Self,
        *exprs: IbisExpr,
        separator: str,
        ignore_nulls: bool,
    ) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            cols = [s for _expr in exprs for s in _expr(df)]
            null_mask = [s.isnull() for _expr in exprs for s in _expr(df)]

            if not ignore_nulls:
                null_mask_result = reduce(operator.or_, null_mask)
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
                result = CaseExpression(
                    condition=~null_mask_result,
                    value=FunctionExpression("concat", *cols_separated),
                )
            else:
                init_value, *values = [
                    CaseExpression(~nm, col.cast(VARCHAR)).otherwise(lit(""))
                    for col, nm in zip(cols, null_mask)
                ]
                separators = (
                    CaseExpression(nm, lit("")).otherwise(lit(separator))
                    for nm in null_mask[:-1]
                )
                result = reduce(
                    lambda x, y: FunctionExpression("concat", x, y),
                    (
                        FunctionExpression("concat", s, v)
                        for s, v in zip(separators, values)
                    ),
                    init_value,
                )

            return [result]

        return IbisExpr(
            call=func,
            function_name="concat_str",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def all_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [reduce(operator.and_, cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def any_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [reduce(operator.or_, cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def max_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            ibis = get_ibis()
            cols = [c for _expr in exprs for c in _expr(df)]
            return [ibis.greatest(*cols).name(cols[0].get_name())]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def min_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            ibis = get_ibis()
            cols = [c for _expr in exprs for c in _expr(df)]
            return [ibis.least(*cols).name(cols[0].get_name())]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def sum_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            cols = [
                e.fill_null(0).name(e.get_name()) for _expr in exprs for e in _expr(df)
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
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            expr = (e.fill_null(0) for _expr in exprs for e in _expr(df))
            non_null = (e.isnull().ifelse(0, 1) for _expr in exprs for e in _expr(df))
            first_expr_name = exprs[0](df)[0].get_name()

            def _name_preserving_sum(e1: ir.Expr, e2: ir.Expr) -> ir.Expr:
                return (e1 + e2).name(e1.get_name())

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

    def when(
        self: Self,
        predicate: IbisExpr,
    ) -> IbisWhen:
        return IbisWhen.from_expr(predicate, context=self)

    def lit(self: Self, value: Any, dtype: DType | type[DType] | None) -> IbisExpr:
        def func(_df: IbisLazyFrame) -> list[ir.Expr]:
            ibis = get_ibis()
            if dtype is not None:
                ibis_dtype = narwhals_to_native_dtype(dtype, version=self._version)
                return [ibis.literal(value, ibis_dtype)]
            return [ibis.literal(value)]

        return self._expr(
            func,
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )

    def len(self: Self) -> IbisExpr:
        def func(_df: IbisLazyFrame) -> list[ir.Expr]:
            return [_df.native.count()]

        return self._expr(
            call=func,
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )


class IbisWhen(LazyWhen["IbisLazyFrame", "ir.Expr", IbisExpr]):
    @property
    def _then(self) -> type[IbisThen]:
        return IbisThen

    def __call__(self: Self, df: IbisLazyFrame) -> Sequence[ir.Expr]:
        ibis = get_ibis()

        self.when = ibis.ifelse
        self.lit = ibis.literal
        return super().__call__(df)


class IbisThen(CompliantThen["IbisLazyFrame", "ir.Expr", IbisExpr], IbisExpr): ...
