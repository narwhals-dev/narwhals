from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Sequence

import duckdb
from duckdb import CaseExpression
from duckdb import CoalesceOperator
from duckdb import FunctionExpression
from duckdb.typing import BIGINT
from duckdb.typing import VARCHAR

from narwhals._compliant import CompliantNamespace
from narwhals._compliant import CompliantThen
from narwhals._compliant import LazyWhen
from narwhals._duckdb.expr import DuckDBExpr
from narwhals._duckdb.selectors import DuckDBSelectorNamespace
from narwhals._duckdb.utils import lit
from narwhals._duckdb.utils import narwhals_to_native_dtype
from narwhals._duckdb.utils import when
from narwhals._expression_parsing import combine_alias_output_names
from narwhals._expression_parsing import combine_evaluate_output_names
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class DuckDBNamespace(CompliantNamespace["DuckDBLazyFrame", "DuckDBExpr"]):
    _implementation: Implementation = Implementation.DUCKDB

    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._version = version

    @property
    def selectors(self: Self) -> DuckDBSelectorNamespace:
        return DuckDBSelectorNamespace(self)

    @property
    def _expr(self) -> type[DuckDBExpr]:
        return DuckDBExpr

    def concat(
        self: Self,
        items: Iterable[DuckDBLazyFrame],
        *,
        how: Literal["horizontal", "vertical", "diagonal"],
    ) -> DuckDBLazyFrame:
        if how == "horizontal":
            msg = "horizontal concat not supported for duckdb. Please join instead"
            raise TypeError(msg)
        if how == "diagonal":
            msg = "Not implemented yet"
            raise NotImplementedError(msg)
        items = list(items)
        first = items[0]
        schema = first.schema
        if how == "vertical" and not all(x.schema == schema for x in items[1:]):
            msg = "inputs should all have the same schema"
            raise TypeError(msg)
        res = reduce(lambda x, y: x.union(y), (item._native_frame for item in items))
        return first._from_native_frame(res)

    def concat_str(
        self: Self,
        *exprs: DuckDBExpr,
        separator: str,
        ignore_nulls: bool,
    ) -> DuckDBExpr:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = [s for _expr in exprs for s in _expr(df)]
            null_mask = [s.isnull() for s in cols]

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

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def all_horizontal(self: Self, *exprs: DuckDBExpr) -> DuckDBExpr:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [reduce(operator.and_, cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def any_horizontal(self: Self, *exprs: DuckDBExpr) -> DuckDBExpr:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [reduce(operator.or_, cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def max_horizontal(self: Self, *exprs: DuckDBExpr) -> DuckDBExpr:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [FunctionExpression("greatest", *cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def min_horizontal(self: Self, *exprs: DuckDBExpr) -> DuckDBExpr:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [FunctionExpression("least", *cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def sum_horizontal(self: Self, *exprs: DuckDBExpr) -> DuckDBExpr:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = (CoalesceOperator(col, lit(0)) for _expr in exprs for col in _expr(df))
            return [reduce(operator.add, cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def mean_horizontal(self: Self, *exprs: DuckDBExpr) -> DuckDBExpr:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = [c for _expr in exprs for c in _expr(df)]
            return [
                (
                    reduce(operator.add, (CoalesceOperator(col, lit(0)) for col in cols))
                    / reduce(operator.add, (col.isnotnull().cast(BIGINT) for col in cols))
                )
            ]

        return DuckDBExpr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def when(self: Self, predicate: DuckDBExpr) -> DuckDBWhen:
        return DuckDBWhen.from_expr(predicate, context=self)

    def lit(self: Self, value: Any, dtype: DType | None) -> DuckDBExpr:
        def func(_df: DuckDBLazyFrame) -> list[duckdb.Expression]:
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

    def len(self: Self) -> DuckDBExpr:
        def func(_df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            return [FunctionExpression("count")]

        return self._expr(
            call=func,
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )


class DuckDBWhen(LazyWhen["DuckDBLazyFrame", duckdb.Expression, DuckDBExpr]):
    @property
    def _then(self) -> type[DuckDBThen]:
        return DuckDBThen

    def __call__(self: Self, df: DuckDBLazyFrame) -> Sequence[duckdb.Expression]:
        self.when = when
        self.lit = lit
        return super().__call__(df)


class DuckDBThen(
    CompliantThen["DuckDBLazyFrame", duckdb.Expression, DuckDBExpr], DuckDBExpr
): ...
