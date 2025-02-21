from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence
from typing import cast

from narwhals._expression_parsing import combine_alias_output_names
from narwhals._expression_parsing import combine_evaluate_output_names
from narwhals._ibis.expr import IbisExpr
from narwhals._ibis.selectors import IbisSelectorNamespace
from narwhals._ibis.utils import ExprKind
from narwhals._ibis.utils import n_ary_operation_expr_kind
from narwhals._ibis.utils import narwhals_to_native_dtype
from narwhals.dependencies import get_ibis
from narwhals.typing import CompliantNamespace

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from typing_extensions import Self

    from narwhals._ibis.dataframe import IbisLazyFrame
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class IbisNamespace(CompliantNamespace["ir.Expr"]):  # type: ignore[type-var]
    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._version = version

    @property
    def selectors(self: Self) -> IbisSelectorNamespace:
        return IbisSelectorNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def all(self: Self) -> IbisExpr:
        def _all(df: IbisLazyFrame) -> list[ir.Expr]:
            ibis = get_ibis()

            return [getattr(ibis._, col_name).name(col_name) for col_name in df.columns]

        return IbisExpr(
            call=_all,
            function_name="all",
            evaluate_output_names=lambda df: df.columns,
            alias_output_names=None,
            expr_kind=ExprKind.TRANSFORM,
            backend_version=self._backend_version,
            version=self._version,
        )

    def concat(
        self: Self,
        items: Sequence[IbisLazyFrame],
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
        first = items[0]
        schema = first.schema
        if how == "vertical" and not all(x.schema == schema for x in items[1:]):
            msg = "inputs should all have the same schema"
            raise TypeError(msg)
        native_dfs = [item._native_frame for item in items]
        res = ibis.union(native_dfs[0], *native_dfs[1:])
        return first._from_native_frame(res)

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

        return IbisExpr(
            call=func,
            function_name="all_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def any_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [reduce(operator.or_, cols)]

        return IbisExpr(
            call=func,
            function_name="or_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def max_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            ibis = get_ibis()
            cols = [c for _expr in exprs for c in _expr(df)]
            return [ibis.greatest(*cols).name(cols[0].get_name())]

        return IbisExpr(
            call=func,
            function_name="max_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def min_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            ibis = get_ibis()
            cols = [c for _expr in exprs for c in _expr(df)]
            return [ibis.least(*cols).name(cols[0].get_name())]

        return IbisExpr(
            call=func,
            function_name="min_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def sum_horizontal(self: Self, *exprs: IbisExpr) -> IbisExpr:
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            cols = [
                e.fill_null(0).name(e.get_name()) for _expr in exprs for e in _expr(df)
            ]
            return [reduce(operator.add, cols)]

        return IbisExpr(
            call=func,
            function_name="sum_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
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

        return IbisExpr(
            call=func,
            function_name="mean_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def when(
        self: Self,
        *predicates: IbisExpr,
    ) -> IbisWhen:
        plx = self.__class__(backend_version=self._backend_version, version=self._version)
        condition = plx.all_horizontal(*predicates)
        return IbisWhen(
            condition,
            self._backend_version,
            expr_kind=ExprKind.TRANSFORM,
            version=self._version,
        )

    def col(self: Self, *column_names: str) -> IbisExpr:
        return IbisExpr.from_column_names(
            *column_names, backend_version=self._backend_version, version=self._version
        )

    def nth(self: Self, *column_indices: int) -> IbisExpr:
        return IbisExpr.from_column_indices(
            *column_indices, backend_version=self._backend_version, version=self._version
        )

    def lit(self: Self, value: Any, dtype: DType | None) -> IbisExpr:
        def func(_df: IbisLazyFrame) -> list[ir.Expr]:
            ibis = get_ibis()
            if dtype is not None:
                ibis_dtype = narwhals_to_native_dtype(dtype, version=self._version)
                return [ibis.literal(value, ibis_dtype)]
            return [ibis.literal(value)]

        return IbisExpr(
            func,
            function_name="lit",
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            expr_kind=ExprKind.LITERAL,
            backend_version=self._backend_version,
            version=self._version,
        )

    def len(self: Self) -> IbisExpr:
        def func(_df: IbisLazyFrame) -> list[ir.Expr]:
            return [_df._native_frame.count()]

        return IbisExpr(
            call=func,
            function_name="len",
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            expr_kind=ExprKind.AGGREGATION,
            backend_version=self._backend_version,
            version=self._version,
        )


class IbisWhen:
    def __init__(
        self: Self,
        condition: IbisExpr,
        backend_version: tuple[int, ...],
        then_value: Any = None,
        otherwise_value: Any = None,
        *,
        expr_kind: ExprKind,
        version: Version,
    ) -> None:
        self._backend_version = backend_version
        self._condition = condition
        self._then_value = then_value
        self._otherwise_value = otherwise_value
        self._expr_kind = expr_kind
        self._version = version

    def __call__(self: Self, df: IbisLazyFrame) -> Sequence[ir.Expr]:
        ibis = get_ibis()

        condition = self._condition(df)[0]
        condition = cast("ir.Expr", condition)

        if isinstance(self._then_value, IbisExpr):
            value = self._then_value(df)[0]
        else:
            # `self._otherwise_value` is a scalar
            value = ibis.literal(self._then_value)
        value = cast("ir.Expr", value)

        if self._otherwise_value is None:
            return [ibis.ifelse(condition, value, None)]
        if not isinstance(self._otherwise_value, IbisExpr):
            # `self._otherwise_value` is a scalar
            otherwise = ibis.literal(self._otherwise_value)
        else:
            otherwise = self._otherwise_value(df)[0]
        return [ibis.ifelse(condition, value, otherwise)]

    def then(self: Self, value: IbisExpr | Any) -> IbisThen:
        self._then_value = value

        return IbisThen(
            self,
            function_name="whenthen",
            evaluate_output_names=getattr(
                value, "_evaluate_output_names", lambda _df: ["literal"]
            ),
            alias_output_names=getattr(value, "_alias_output_names", None),
            expr_kind=self._expr_kind,
            backend_version=self._backend_version,
            version=self._version,
        )


class IbisThen(IbisExpr):
    def __init__(
        self: Self,
        call: IbisWhen,
        *,
        function_name: str,
        evaluate_output_names: Callable[[IbisLazyFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        expr_kind: ExprKind,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._backend_version = backend_version
        self._version = version
        self._call = call
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._expr_kind = expr_kind

    def otherwise(self: Self, value: IbisExpr | Any) -> IbisExpr:
        # type ignore because we are setting the `_call` attribute to a
        # callable object of type `DuckDBWhen`, base class has the attribute as
        # only a `Callable`
        self._call._otherwise_value = value  # type: ignore[attr-defined]
        self._function_name = "whenotherwise"
        return self
