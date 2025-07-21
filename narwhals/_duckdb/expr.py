from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

from duckdb import CoalesceOperator, StarExpression
from duckdb.typing import DuckDBPyType

from narwhals._duckdb.expr_dt import DuckDBExprDateTimeNamespace
from narwhals._duckdb.expr_list import DuckDBExprListNamespace
from narwhals._duckdb.expr_str import DuckDBExprStringNamespace
from narwhals._duckdb.expr_struct import DuckDBExprStructNamespace
from narwhals._duckdb.utils import (
    DeferredTimeZone,
    F,
    col,
    lit,
    narwhals_to_native_dtype,
    when,
    window_expression,
)
from narwhals._expression_parsing import ExprKind, ExprMetadata
from narwhals._sql.expr import SQLExpr
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckdb import Expression
    from typing_extensions import Self

    from narwhals._compliant import WindowInputs
    from narwhals._compliant.typing import (
        AliasNames,
        EvalNames,
        EvalSeries,
        WindowFunction,
    )
    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.namespace import DuckDBNamespace
    from narwhals._utils import _LimitedContext
    from narwhals.typing import (
        FillNullStrategy,
        IntoDType,
        NonNestedLiteral,
        NumericLiteral,
        RollingInterpolationMethod,
        TemporalLiteral,
    )

    DuckDBWindowFunction = WindowFunction[DuckDBLazyFrame, Expression]
    DuckDBWindowInputs = WindowInputs[Expression]


class DuckDBExpr(SQLExpr["DuckDBLazyFrame", "Expression"]):
    _implementation = Implementation.DUCKDB

    def __init__(
        self,
        call: EvalSeries[DuckDBLazyFrame, Expression],
        window_function: DuckDBWindowFunction | None = None,
        *,
        evaluate_output_names: EvalNames[DuckDBLazyFrame],
        alias_output_names: AliasNames | None,
        version: Version,
        implementation: Implementation = Implementation.DUCKDB,
    ) -> None:
        self._call = call
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._version = version
        self._metadata: ExprMetadata | None = None
        self._window_function: DuckDBWindowFunction | None = window_function

    def _function(self, name: str, *args: Expression) -> Expression:  # type: ignore[override]
        if name == "isnull":
            return args[0].isnull()
        return F(name, *args)

    def _lit(self, value: Any) -> Expression:
        return lit(value)

    def _count_star(self) -> Expression:
        return F("count", StarExpression())

    def _when(self, condition: Expression, value: Expression) -> Expression:
        return when(condition, value)

    def _window_expression(
        self,
        expr: Expression,
        partition_by: Sequence[str | Expression] = (),
        order_by: Sequence[str | Expression] = (),
        rows_start: int | None = None,
        rows_end: int | None = None,
        *,
        descending: Sequence[bool] | None = None,
        nulls_last: Sequence[bool] | None = None,
    ) -> Expression:
        return window_expression(
            expr,
            partition_by,
            order_by,
            rows_start,
            rows_end,
            descending=descending,
            nulls_last=nulls_last,
        )

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> DuckDBNamespace:  # pragma: no cover
        from narwhals._duckdb.namespace import DuckDBNamespace

        return DuckDBNamespace(version=self._version)

    def broadcast(self, kind: Literal[ExprKind.AGGREGATION, ExprKind.LITERAL]) -> Self:
        if kind is ExprKind.LITERAL:
            return self
        if self._backend_version < (1, 3):
            msg = "At least version 1.3 of DuckDB is required for binary operations between aggregates and columns."
            raise NotImplementedError(msg)
        return self.over([lit(1)], [])

    @classmethod
    def from_column_names(
        cls,
        evaluate_column_names: EvalNames[DuckDBLazyFrame],
        /,
        *,
        context: _LimitedContext,
    ) -> Self:
        def func(df: DuckDBLazyFrame) -> list[Expression]:
            return [col(name) for name in evaluate_column_names(df)]

        return cls(
            func,
            evaluate_output_names=evaluate_column_names,
            alias_output_names=None,
            version=context._version,
        )

    @classmethod
    def from_column_indices(cls, *column_indices: int, context: _LimitedContext) -> Self:
        def func(df: DuckDBLazyFrame) -> list[Expression]:
            columns = df.columns
            return [col(columns[i]) for i in column_indices]

        return cls(
            func,
            evaluate_output_names=cls._eval_names_indices(column_indices),
            alias_output_names=None,
            version=context._version,
        )

    @classmethod
    def _alias_native(cls, expr: Expression, name: str) -> Expression:
        return expr.alias(name)

    def __invert__(self) -> Self:
        invert = cast("Callable[..., Expression]", operator.invert)
        return self._with_elementwise(invert)

    def skew(self) -> Self:
        def func(expr: Expression) -> Expression:
            count = F("count", expr)
            # Adjust population skewness by correction factor to get sample skewness
            sample_skewness = (
                F("skewness", expr)
                * (count - lit(2))
                / F("sqrt", count * (count - lit(1)))
            )
            return when(count == lit(0), lit(None)).otherwise(
                when(count == lit(1), lit(float("nan"))).otherwise(
                    when(count == lit(2), lit(0.0)).otherwise(sample_skewness)
                )
            )

        return self._with_callable(func)

    def kurtosis(self) -> Self:
        return self._with_callable(lambda expr: F("kurtosis_pop", expr))

    def all(self) -> Self:
        def f(expr: Expression) -> Expression:
            return CoalesceOperator(F("bool_and", expr), lit(True))  # noqa: FBT003

        def window_f(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> list[Expression]:
            return [
                CoalesceOperator(
                    window_expression(F("bool_and", expr), inputs.partition_by),
                    lit(True),  # noqa: FBT003
                )
                for expr in self(df)
            ]

        return self._with_callable(f)._with_window_function(window_f)

    def any(self) -> Self:
        def f(expr: Expression) -> Expression:
            return CoalesceOperator(F("bool_or", expr), lit(False))  # noqa: FBT003

        def window_f(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> list[Expression]:
            return [
                CoalesceOperator(
                    window_expression(F("bool_or", expr), inputs.partition_by),
                    lit(False),  # noqa: FBT003
                )
                for expr in self(df)
            ]

        return self._with_callable(f)._with_window_function(window_f)

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod
    ) -> Self:
        def func(expr: Expression) -> Expression:
            if interpolation == "linear":
                return F("quantile_cont", expr, lit(quantile))
            msg = "Only linear interpolation methods are supported for DuckDB quantile."
            raise NotImplementedError(msg)

        return self._with_callable(func)

    def clip(
        self,
        lower_bound: Self | NumericLiteral | TemporalLiteral | None,
        upper_bound: Self | NumericLiteral | TemporalLiteral | None,
    ) -> Self:
        def _clip_lower(expr: Expression, lower_bound: Any) -> Expression:
            return F("greatest", expr, lower_bound)

        def _clip_upper(expr: Expression, upper_bound: Any) -> Expression:
            return F("least", expr, upper_bound)

        def _clip_both(
            expr: Expression, lower_bound: Any, upper_bound: Any
        ) -> Expression:
            return F("greatest", F("least", expr, upper_bound), lower_bound)

        if lower_bound is None:
            return self._with_elementwise(_clip_upper, upper_bound=upper_bound)
        if upper_bound is None:
            return self._with_elementwise(_clip_lower, lower_bound=lower_bound)
        return self._with_elementwise(
            _clip_both, lower_bound=lower_bound, upper_bound=upper_bound
        )

    def sum(self) -> Self:
        def f(expr: Expression) -> Expression:
            return CoalesceOperator(F("sum", expr), lit(0))

        def window_f(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> list[Expression]:
            return [
                CoalesceOperator(
                    window_expression(F("sum", expr), inputs.partition_by), lit(0)
                )
                for expr in self(df)
            ]

        return self._with_callable(f)._with_window_function(window_f)

    def n_unique(self) -> Self:
        def func(expr: Expression) -> Expression:
            # https://stackoverflow.com/a/79338887/4451315
            return F("array_unique", F("array_agg", expr)) + F(
                "max", when(expr.isnotnull(), lit(0)).otherwise(lit(1))
            )

        return self._with_callable(func)

    def count(self) -> Self:
        return self._with_callable(lambda expr: F("count", expr))

    def len(self) -> Self:
        return self._with_callable(lambda _expr: F("count"))

    def std(self, ddof: int) -> Self:
        if ddof == 0:
            return self._with_callable(lambda expr: F("stddev_pop", expr))
        if ddof == 1:
            return self._with_callable(lambda expr: F("stddev_samp", expr))

        def _std(expr: Expression) -> Expression:
            n_samples = F("count", expr)
            return (
                F("stddev_pop", expr)
                * F("sqrt", n_samples)
                / (F("sqrt", (n_samples - lit(ddof))))
            )

        return self._with_callable(_std)

    def var(self, ddof: int) -> Self:
        if ddof == 0:
            return self._with_callable(lambda expr: F("var_pop", expr))
        if ddof == 1:
            return self._with_callable(lambda expr: F("var_samp", expr))

        def _var(expr: Expression) -> Expression:
            n_samples = F("count", expr)
            return F("var_pop", expr) * n_samples / (n_samples - lit(ddof))

        return self._with_callable(_var)

    def null_count(self) -> Self:
        return self._with_callable(lambda expr: F("sum", expr.isnull().cast("int")))

    def is_null(self) -> Self:
        return self._with_elementwise(lambda expr: expr.isnull())

    def is_nan(self) -> Self:
        return self._with_elementwise(lambda expr: F("isnan", expr))

    def is_finite(self) -> Self:
        return self._with_elementwise(lambda expr: F("isfinite", expr))

    def is_in(self, other: Sequence[Any]) -> Self:
        return self._with_elementwise(lambda expr: F("contains", lit(other), expr))

    def fill_null(
        self,
        value: Self | NonNestedLiteral,
        strategy: FillNullStrategy | None,
        limit: int | None,
    ) -> Self:
        if strategy is not None:
            if self._backend_version < (1, 3):  # pragma: no cover
                msg = f"`fill_null` with `strategy={strategy}` is only available in 'duckdb>=1.3.0'."
                raise NotImplementedError(msg)

            def _fill_with_strategy(
                df: DuckDBLazyFrame, inputs: DuckDBWindowInputs
            ) -> Sequence[Expression]:
                fill_func = "last_value" if strategy == "forward" else "first_value"
                rows_start, rows_end = (
                    (-limit if limit is not None else None, 0)
                    if strategy == "forward"
                    else (0, limit)
                )
                return [
                    window_expression(
                        F(fill_func, expr),
                        inputs.partition_by,
                        inputs.order_by,
                        rows_start=rows_start,
                        rows_end=rows_end,
                        ignore_nulls=True,
                    )
                    for expr in self(df)
                ]

            return self._with_window_function(_fill_with_strategy)

        def _fill_constant(expr: Expression, value: Any) -> Expression:
            return CoalesceOperator(expr, value)

        return self._with_elementwise(_fill_constant, value=value)

    def cast(self, dtype: IntoDType) -> Self:
        def func(df: DuckDBLazyFrame) -> list[Expression]:
            tz = DeferredTimeZone(df.native)
            native_dtype = narwhals_to_native_dtype(dtype, self._version, tz)
            return [expr.cast(DuckDBPyType(native_dtype)) for expr in self(df)]

        def window_f(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> list[Expression]:
            tz = DeferredTimeZone(df.native)
            native_dtype = narwhals_to_native_dtype(dtype, self._version, tz)
            return [
                expr.cast(DuckDBPyType(native_dtype))
                for expr in self.window_function(df, inputs)
            ]

        return self.__class__(
            func,
            window_f,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def log(self, base: float) -> Self:
        def _log(expr: Expression) -> Expression:
            log = F("log", expr)
            return (
                when(expr < lit(0), lit(float("nan")))
                .when(expr == lit(0), lit(float("-inf")))
                .otherwise(log / F("log", lit(base)))
            )

        return self._with_elementwise(_log)

    def exp(self) -> Self:
        def _exp(expr: Expression) -> Expression:
            return F("exp", expr)

        return self._with_elementwise(_exp)

    def sqrt(self) -> Self:
        def _sqrt(expr: Expression) -> Expression:
            return when(expr < lit(0), lit(float("nan"))).otherwise(F("sqrt", expr))

        return self._with_elementwise(_sqrt)

    @property
    def str(self) -> DuckDBExprStringNamespace:
        return DuckDBExprStringNamespace(self)

    @property
    def dt(self) -> DuckDBExprDateTimeNamespace:
        return DuckDBExprDateTimeNamespace(self)

    @property
    def list(self) -> DuckDBExprListNamespace:
        return DuckDBExprListNamespace(self)

    @property
    def struct(self) -> DuckDBExprStructNamespace:
        return DuckDBExprStructNamespace(self)
