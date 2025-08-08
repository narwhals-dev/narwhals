from __future__ import annotations

import operator
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Any, cast

import ibis
import ibis.expr.types as ir

from narwhals._expression_parsing import (
    combine_alias_output_names,
    combine_evaluate_output_names,
)
from narwhals._ibis.dataframe import IbisLazyFrame
from narwhals._ibis.expr import IbisExpr
from narwhals._ibis.selectors import IbisSelectorNamespace
from narwhals._ibis.utils import function, lit, narwhals_to_native_dtype
from narwhals._sql.namespace import SQLNamespace
from narwhals._sql.when_then import SQLThen, SQLWhen
from narwhals._utils import Implementation, requires

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ibis import Deferred

    from narwhals._utils import Version
    from narwhals.typing import ConcatMethod, IntoDType, PythonLiteral


class IbisNamespace(SQLNamespace[IbisLazyFrame, IbisExpr, "ir.Table", "Deferred"]):
    _implementation: Implementation = Implementation.IBIS

    def __init__(self, *, version: Version) -> None:
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

    def _function(self, name: str, *args: Deferred | PythonLiteral) -> Deferred:
        return function(name, *args)

    def _lit(self, value: Any) -> Deferred:
        return lit(value)  # pyright: ignore[reportReturnType]

    def _when(
        self, condition: Deferred, value: Deferred, otherwise: ir.Deferred | None = None
    ) -> ir.Deferred:
        if otherwise is None:
            return ibis.cases((condition, value))  # pyright: ignore[reportReturnType]
        return ibis.cases(
            (condition, value), else_=otherwise
        )  # pragma: no cover  # pyright: ignore[reportReturnType]

    def _coalesce(self, *exprs: ir.Deferred) -> ir.Deferred:
        return ibis.coalesce(*exprs)  # pyright: ignore[reportReturnType]

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
        def func(df: IbisLazyFrame) -> list[ir.Deferred]:
            cols = list(chain.from_iterable(expr(df) for expr in exprs))
            cols_casted = [s.cast("string") for s in cols]

            if not ignore_nulls:
                result = cols_casted[0]
                for col in cols_casted[1:]:
                    result = result + separator + col
            else:
                sep = cast("ir.Deferred", lit(separator))
                result = sep.join(cols_casted)

            return [result]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            version=self._version,
        )

    def mean_horizontal(self, *exprs: IbisExpr) -> IbisExpr:
        def func(cols: Iterable[ir.Deferred]) -> ir.Deferred:
            cols = list(cols)
            return reduce(operator.add, (col.fill_null(lit(0)) for col in cols)) / reduce(
                operator.add, (col.isnull().ifelse(lit(0), lit(1)) for col in cols)
            )

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    @requires.backend_version((10, 0))
    def when(self, predicate: IbisExpr) -> IbisWhen:
        return IbisWhen.from_expr(predicate, context=self)

    def lit(self, value: Any, dtype: IntoDType | None) -> IbisExpr:
        def func(_df: IbisLazyFrame) -> Sequence[ir.Deferred]:
            ibis_dtype = narwhals_to_native_dtype(dtype, self._version) if dtype else None
            return [lit(value, ibis_dtype)]  # pyright: ignore[reportReturnType]

        return self._expr(
            func,
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            version=self._version,
        )

    def len(self) -> IbisExpr:
        def func(_df: IbisLazyFrame) -> list[ir.Deferred]:
            return [_df.native.count()]  # pyright: ignore[reportReturnType]

        return self._expr(
            call=func,
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            version=self._version,
        )


class IbisWhen(SQLWhen["IbisLazyFrame", "ir.Deferred", IbisExpr]):
    lit = lit

    @property
    def _then(self) -> type[IbisThen]:
        return IbisThen

    def __call__(self, df: IbisLazyFrame) -> Sequence[ir.Deferred]:
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
        return [result]  # pyright: ignore[reportReturnType]


class IbisThen(SQLThen["IbisLazyFrame", "ir.Deferred", IbisExpr], IbisExpr): ...
