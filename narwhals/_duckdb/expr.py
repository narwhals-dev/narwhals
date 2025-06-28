from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

from duckdb import CoalesceOperator, StarExpression
from duckdb.typing import DuckDBPyType

from narwhals._compliant import LazyExpr
from narwhals._compliant.window import WindowInputs
from narwhals._duckdb.expr_dt import DuckDBExprDateTimeNamespace
from narwhals._duckdb.expr_list import DuckDBExprListNamespace
from narwhals._duckdb.expr_str import DuckDBExprStringNamespace
from narwhals._duckdb.expr_struct import DuckDBExprStructNamespace
from narwhals._duckdb.utils import (
    F,
    col,
    lit,
    narwhals_to_native_dtype,
    when,
    window_expression,
)
from narwhals._expression_parsing import (
    ExprKind,
    combine_alias_output_names,
    combine_evaluate_output_names,
)
from narwhals._utils import Implementation, not_implemented, requires

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from duckdb import Expression
    from typing_extensions import Self

    from narwhals._compliant.typing import (
        AliasNames,
        EvalNames,
        EvalSeries,
        WindowFunction,
    )
    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.namespace import DuckDBNamespace
    from narwhals._duckdb.typing import WindowExpressionKwargs
    from narwhals._expression_parsing import ExprMetadata
    from narwhals._utils import Version, _FullContext
    from narwhals.typing import (
        FillNullStrategy,
        IntoDType,
        NonNestedLiteral,
        NumericLiteral,
        RankMethod,
        RollingInterpolationMethod,
        TemporalLiteral,
    )

    DuckDBWindowFunction = WindowFunction[DuckDBLazyFrame, Expression]
    DuckDBWindowInputs = WindowInputs[Expression]


class DuckDBExpr(LazyExpr["DuckDBLazyFrame", "Expression"]):
    _implementation = Implementation.DUCKDB

    def __init__(
        self,
        call: EvalSeries[DuckDBLazyFrame, Expression],
        window_function: DuckDBWindowFunction | None = None,
        *,
        evaluate_output_names: EvalNames[DuckDBLazyFrame],
        alias_output_names: AliasNames | None,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._call = call
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._backend_version = backend_version
        self._version = version
        self._metadata: ExprMetadata | None = None
        self._window_function: DuckDBWindowFunction | None = window_function

    @property
    def window_function(self) -> DuckDBWindowFunction:
        def default_window_func(
            df: DuckDBLazyFrame, inputs: DuckDBWindowInputs
        ) -> list[Expression]:
            assert not inputs.order_by  # noqa: S101
            return [
                window_expression(expr, inputs.partition_by, inputs.order_by)
                for expr in self(df)
            ]

        return self._window_function or default_window_func

    def __call__(self, df: DuckDBLazyFrame) -> Sequence[Expression]:
        return self._call(df)

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> DuckDBNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._duckdb.namespace import DuckDBNamespace

        return DuckDBNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def _cum_window_func(
        self,
        func_name: Literal["sum", "max", "min", "count", "product"],
        *,
        reverse: bool,
    ) -> DuckDBWindowFunction:
        def func(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> list[Expression]:
            return [
                window_expression(
                    F(func_name, expr),
                    inputs.partition_by,
                    inputs.order_by,
                    descending=reverse,
                    nulls_last=reverse,
                    rows_start="unbounded preceding",
                    rows_end="current row",
                )
                for expr in self(df)
            ]

        return func

    def _rolling_window_func(
        self,
        func_name: Literal["sum", "mean", "std", "var"],
        window_size: int,
        min_samples: int,
        ddof: int | None = None,
        *,
        center: bool,
    ) -> DuckDBWindowFunction:
        supported_funcs = ["sum", "mean", "std", "var"]
        if center:
            half = (window_size - 1) // 2
            remainder = (window_size - 1) % 2
            start = f"{half + remainder} preceding"
            end = f"{half} following"
        else:
            start = f"{window_size - 1} preceding"
            end = "current row"

        def func(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> list[Expression]:
            if func_name in {"sum", "mean"}:
                func_: str = func_name
            elif func_name == "var" and ddof == 0:
                func_ = "var_pop"
            elif func_name in "var" and ddof == 1:
                func_ = "var_samp"
            elif func_name == "std" and ddof == 0:
                func_ = "stddev_pop"
            elif func_name == "std" and ddof == 1:
                func_ = "stddev_samp"
            elif func_name in {"var", "std"}:  # pragma: no cover
                msg = f"Only ddof=0 and ddof=1 are currently supported for rolling_{func_name}."
                raise ValueError(msg)
            else:  # pragma: no cover
                msg = f"Only the following functions are supported: {supported_funcs}.\nGot: {func_name}."
                raise ValueError(msg)
            window_kwargs: WindowExpressionKwargs = {
                "partition_by": inputs.partition_by,
                "order_by": inputs.order_by,
                "rows_start": start,
                "rows_end": end,
            }
            return [
                when(
                    window_expression(F("count", expr), **window_kwargs)
                    >= lit(min_samples),
                    window_expression(F(func_, expr), **window_kwargs),
                )
                for expr in self(df)
            ]

        return func

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
        context: _FullContext,
    ) -> Self:
        def func(df: DuckDBLazyFrame) -> list[Expression]:
            return [col(name) for name in evaluate_column_names(df)]

        return cls(
            func,
            evaluate_output_names=evaluate_column_names,
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
        )

    @classmethod
    def from_column_indices(cls, *column_indices: int, context: _FullContext) -> Self:
        def func(df: DuckDBLazyFrame) -> list[Expression]:
            columns = df.columns
            return [col(columns[i]) for i in column_indices]

        return cls(
            func,
            evaluate_output_names=cls._eval_names_indices(column_indices),
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
        )

    @classmethod
    def _from_elementwise_horizontal_op(
        cls, func: Callable[[Iterable[Expression]], Expression], *exprs: Self
    ) -> Self:
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

        context = exprs[0]
        return cls(
            call=call,
            window_function=window_function,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=context._backend_version,
            version=context._version,
        )

    def _callable_to_eval_series(
        self, call: Callable[..., Expression], /, **expressifiable_args: Self | Any
    ) -> EvalSeries[DuckDBLazyFrame, Expression]:
        def func(df: DuckDBLazyFrame) -> list[Expression]:
            native_series_list = self(df)
            other_native_series = {
                key: df._evaluate_expr(value) if self._is_expr(value) else lit(value)
                for key, value in expressifiable_args.items()
            }
            return [
                call(native_series, **other_native_series)
                for native_series in native_series_list
            ]

        return func

    def _push_down_window_function(
        self, call: Callable[..., Expression], /, **expressifiable_args: Self | Any
    ) -> DuckDBWindowFunction:
        def window_f(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> list[Expression]:
            # If a function `f` is elementwise, and `g` is another function, then
            # - `f(g) over (window)`
            # - `f(g over (window))
            # are equivalent.
            # Make sure to only use with if `call` is elementwise!
            native_series_list = self.window_function(df, inputs)
            other_native_series = {
                key: df._evaluate_window_expr(value, inputs)
                if self._is_expr(value)
                else lit(value)
                for key, value in expressifiable_args.items()
            }
            return [
                call(native_series, **other_native_series)
                for native_series in native_series_list
            ]

        return window_f

    def _with_callable(
        self, call: Callable[..., Expression], /, **expressifiable_args: Self | Any
    ) -> Self:
        """Create expression from callable.

        Arguments:
            call: Callable from compliant DataFrame to native Expression
            expr_name: Expression name
            expressifiable_args: arguments pass to expression which should be parsed
                as expressions (e.g. in `nw.col('a').is_between('b', 'c')`)
        """
        return self.__class__(
            self._callable_to_eval_series(call, **expressifiable_args),
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def _with_elementwise(
        self, call: Callable[..., Expression], /, **expressifiable_args: Self | Any
    ) -> Self:
        return self.__class__(
            self._callable_to_eval_series(call, **expressifiable_args),
            self._push_down_window_function(call, **expressifiable_args),
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def _with_binary(self, op: Callable[..., Expression], other: Self | Any) -> Self:
        return self.__class__(
            self._callable_to_eval_series(op, other=other),
            self._push_down_window_function(op, other=other),
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def _with_alias_output_names(self, func: AliasNames | None, /) -> Self:
        return type(self)(
            self._call,
            self._window_function,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=func,
            backend_version=self._backend_version,
            version=self._version,
        )

    def _with_window_function(self, window_function: DuckDBWindowFunction) -> Self:
        return self.__class__(
            self._call,
            window_function,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    @classmethod
    def _alias_native(cls, expr: Expression, name: str) -> Expression:
        return expr.alias(name)

    def __invert__(self) -> Self:
        invert = cast("Callable[..., Expression]", operator.invert)
        return self._with_elementwise(invert)

    def abs(self) -> Self:
        return self._with_elementwise(lambda expr: F("abs", expr))

    def mean(self) -> Self:
        return self._with_callable(lambda expr: F("mean", expr))

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

    def median(self) -> Self:
        return self._with_callable(lambda expr: F("median", expr))

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

    def max(self) -> Self:
        return self._with_callable(lambda expr: F("max", expr))

    def min(self) -> Self:
        return self._with_callable(lambda expr: F("min", expr))

    def null_count(self) -> Self:
        return self._with_callable(lambda expr: F("sum", expr.isnull().cast("int")))

    @requires.backend_version((1, 3))
    def over(
        self, partition_by: Sequence[str | Expression], order_by: Sequence[str]
    ) -> Self:
        def func(df: DuckDBLazyFrame) -> Sequence[Expression]:
            return self.window_function(df, WindowInputs(partition_by, order_by))

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def is_null(self) -> Self:
        return self._with_elementwise(lambda expr: expr.isnull())

    def is_nan(self) -> Self:
        return self._with_elementwise(lambda expr: F("isnan", expr))

    def is_finite(self) -> Self:
        return self._with_elementwise(lambda expr: F("isfinite", expr))

    def is_in(self, other: Sequence[Any]) -> Self:
        return self._with_elementwise(lambda expr: F("contains", lit(other), expr))

    def round(self, decimals: int) -> Self:
        return self._with_elementwise(lambda expr: F("round", expr, lit(decimals)))

    @requires.backend_version((1, 3))
    def shift(self, n: int) -> Self:
        def func(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> Sequence[Expression]:
            return [
                window_expression(
                    F("lag", expr, lit(n)), inputs.partition_by, inputs.order_by
                )
                for expr in self(df)
            ]

        return self._with_window_function(func)

    @requires.backend_version((1, 3))
    def is_first_distinct(self) -> Self:
        def func(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> Sequence[Expression]:
            return [
                window_expression(
                    F("row_number"), (*inputs.partition_by, expr), inputs.order_by
                )
                == lit(1)
                for expr in self(df)
            ]

        return self._with_window_function(func)

    @requires.backend_version((1, 3))
    def is_last_distinct(self) -> Self:
        def func(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> Sequence[Expression]:
            return [
                window_expression(
                    F("row_number"),
                    (*inputs.partition_by, expr),
                    inputs.order_by,
                    descending=True,
                    nulls_last=True,
                )
                == lit(1)
                for expr in self(df)
            ]

        return self._with_window_function(func)

    @requires.backend_version((1, 3))
    def diff(self) -> Self:
        def func(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> list[Expression]:
            return [
                expr
                - window_expression(F("lag", expr), inputs.partition_by, inputs.order_by)
                for expr in self(df)
            ]

        return self._with_window_function(func)

    @requires.backend_version((1, 3))
    def cum_sum(self, *, reverse: bool) -> Self:
        return self._with_window_function(self._cum_window_func("sum", reverse=reverse))

    @requires.backend_version((1, 3))
    def cum_max(self, *, reverse: bool) -> Self:
        return self._with_window_function(self._cum_window_func("max", reverse=reverse))

    @requires.backend_version((1, 3))
    def cum_min(self, *, reverse: bool) -> Self:
        return self._with_window_function(self._cum_window_func("min", reverse=reverse))

    @requires.backend_version((1, 3))
    def cum_count(self, *, reverse: bool) -> Self:
        return self._with_window_function(self._cum_window_func("count", reverse=reverse))

    @requires.backend_version((1, 3))
    def cum_prod(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func("product", reverse=reverse)
        )

    @requires.backend_version((1, 3))
    def rolling_sum(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._with_window_function(
            self._rolling_window_func("sum", window_size, min_samples, center=center)
        )

    @requires.backend_version((1, 3))
    def rolling_mean(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._with_window_function(
            self._rolling_window_func("mean", window_size, min_samples, center=center)
        )

    @requires.backend_version((1, 3))
    def rolling_var(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                "var", window_size, min_samples, ddof=ddof, center=center
            )
        )

    @requires.backend_version((1, 3))
    def rolling_std(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                "std", window_size, min_samples, ddof=ddof, center=center
            )
        )

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
                _limit = "unbounded" if limit is None else limit
                rows_start, rows_end = (
                    (f"{_limit} preceding", "current row")
                    if strategy == "forward"
                    else ("current row", f"{_limit} following")
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
        def func(expr: Expression) -> Expression:
            native_dtype = narwhals_to_native_dtype(dtype, self._version)
            return expr.cast(DuckDBPyType(native_dtype))

        return self._with_elementwise(func)

    @requires.backend_version((1, 3))
    def is_unique(self) -> Self:
        def _is_unique(expr: Expression, *partition_by: str | Expression) -> Expression:
            return window_expression(
                F("count", StarExpression()), (expr, *partition_by)
            ) == lit(1)

        def _unpartitioned_is_unique(expr: Expression) -> Expression:
            return _is_unique(expr)

        def _partitioned_is_unique(
            df: DuckDBLazyFrame, inputs: DuckDBWindowInputs
        ) -> Sequence[Expression]:
            assert not inputs.order_by  # noqa: S101
            return [_is_unique(expr, *inputs.partition_by) for expr in self(df)]

        return self._with_callable(_unpartitioned_is_unique)._with_window_function(
            _partitioned_is_unique
        )

    @requires.backend_version((1, 3))
    def rank(self, method: RankMethod, *, descending: bool) -> Self:
        if method in {"min", "max", "average"}:
            func = F("rank")
        elif method == "dense":
            func = F("dense_rank")
        else:  # method == "ordinal"
            func = F("row_number")

        def _rank(
            expr: Expression,
            *,
            descending: bool,
            partition_by: Sequence[str | Expression],
        ) -> Expression:
            count_expr = F("count", StarExpression())
            window_kwargs: WindowExpressionKwargs = {
                "partition_by": partition_by,
                "order_by": (expr,),
                "descending": descending,
                "nulls_last": True,
            }
            count_window_kwargs: WindowExpressionKwargs = {
                "partition_by": (*partition_by, expr)
            }
            if method == "max":
                rank_expr = (
                    window_expression(func, **window_kwargs)
                    + window_expression(count_expr, **count_window_kwargs)
                    - lit(1)
                )
            elif method == "average":
                rank_expr = window_expression(func, **window_kwargs) + (
                    window_expression(count_expr, **count_window_kwargs) - lit(1)
                ) / lit(2.0)
            else:
                rank_expr = window_expression(func, **window_kwargs)
            return when(expr.isnotnull(), rank_expr)

        def _unpartitioned_rank(expr: Expression) -> Expression:
            return _rank(expr, partition_by=(), descending=descending)

        def _partitioned_rank(
            df: DuckDBLazyFrame, inputs: DuckDBWindowInputs
        ) -> Sequence[Expression]:
            assert not inputs.order_by  # noqa: S101
            return [
                _rank(expr, descending=descending, partition_by=inputs.partition_by)
                for expr in self(df)
            ]

        return self._with_callable(_unpartitioned_rank)._with_window_function(
            _partitioned_rank
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

    drop_nulls = not_implemented()
    unique = not_implemented()
