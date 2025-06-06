from __future__ import annotations

import contextlib
import operator
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, cast

from duckdb import CoalesceOperator, FunctionExpression, StarExpression
from duckdb.typing import DuckDBPyType

from narwhals._compliant import LazyExpr
from narwhals._compliant.window import WindowInputs
from narwhals._duckdb.expr_dt import DuckDBExprDateTimeNamespace
from narwhals._duckdb.expr_list import DuckDBExprListNamespace
from narwhals._duckdb.expr_str import DuckDBExprStringNamespace
from narwhals._duckdb.expr_struct import DuckDBExprStructNamespace
from narwhals._duckdb.utils import (
    col,
    generate_order_by_sql,
    generate_partition_by_sql,
    lit,
    narwhals_to_native_dtype,
    when,
)
from narwhals._expression_parsing import ExprKind
from narwhals.utils import Implementation, not_implemented, requires

if TYPE_CHECKING:
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
    from narwhals._expression_parsing import ExprMetadata
    from narwhals.dtypes import DType
    from narwhals.typing import (
        FillNullStrategy,
        NonNestedLiteral,
        NumericLiteral,
        RankMethod,
        RollingInterpolationMethod,
        TemporalLiteral,
    )
    from narwhals.utils import Version, _FullContext

    DuckDBWindowInputs = WindowInputs[Expression]
    DuckDBWindowFunction = WindowFunction[DuckDBLazyFrame, Expression]


with contextlib.suppress(ImportError):  # requires duckdb>=1.3.0
    from duckdb import SQLExpression


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

        if window_function is not None:
            self._window_function: DuckDBWindowFunction = window_function
        else:

            def window_func(
                df: DuckDBLazyFrame, window_inputs: DuckDBWindowInputs
            ) -> list[Expression]:
                assert not window_inputs.order_by  # noqa: S101
                partition_by_sql = generate_partition_by_sql(*window_inputs.partition_by)
                template = f"{{expr}} over ({partition_by_sql})"
                return [SQLExpression(template.format(expr=expr)) for expr in self(df)]

            self._window_function = window_func

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
        *,
        reverse: bool,
        func_name: Literal["sum", "max", "min", "count", "product"],
    ) -> DuckDBWindowFunction:
        def func(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> list[Expression]:
            order_by_sql = generate_order_by_sql(*inputs.order_by, ascending=not reverse)
            partition_by_sql = generate_partition_by_sql(*inputs.partition_by)
            sql = (
                f"{func_name} ({{expr}}) over ({partition_by_sql} {order_by_sql} "
                "rows between unbounded preceding and current row)"
            )
            return [SQLExpression(sql.format(expr=expr)) for expr in self(df)]

        return func

    def _rolling_window_func(
        self,
        *,
        func_name: Literal["sum", "mean", "std", "var"],
        center: bool,
        window_size: int,
        min_samples: int,
        ddof: int | None = None,
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
            order_by_sql = generate_order_by_sql(*inputs.order_by, ascending=True)
            partition_by_sql = generate_partition_by_sql(*inputs.partition_by)
            window = f"({partition_by_sql} {order_by_sql} rows between {start} and {end})"
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
            condition_sql = f"count({{expr}}) over {window} >= {min_samples}"
            value_sql = f"{func_}({{expr}}) over {window}"
            return [
                when(
                    SQLExpression(condition_sql.format(expr=expr)),
                    SQLExpression(value_sql.format(expr=expr)),
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
        template = "{expr} over ()"
        return self._with_callable(lambda expr: SQLExpression(template.format(expr=expr)))

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

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def _with_alias_output_names(self, func: AliasNames | None, /) -> Self:
        return type(self)(
            call=self._call,
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

    def __and__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda expr, other: expr & other, other=other)

    def __or__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda expr, other: expr | other, other=other)

    def __add__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda expr, other: expr + other, other=other)

    def __truediv__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda expr, other: expr / other, other=other)

    def __rtruediv__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(
            lambda expr, other: other.__truediv__(expr), other=other
        ).alias("literal")

    def __floordiv__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(
            lambda expr, other: expr.__floordiv__(other), other=other
        )

    def __rfloordiv__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(
            lambda expr, other: other.__floordiv__(expr), other=other
        ).alias("literal")

    def __mod__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda expr, other: expr.__mod__(other), other=other)

    def __rmod__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(
            lambda expr, other: other.__mod__(expr), other=other
        ).alias("literal")

    def __sub__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda expr, other: expr - other, other=other)

    def __rsub__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(
            lambda expr, other: other.__sub__(expr), other=other
        ).alias("literal")

    def __mul__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda expr, other: expr * other, other=other)

    def __pow__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda expr, other: expr**other, other=other)

    def __rpow__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(
            lambda expr, other: other.__pow__(expr), other=other
        ).alias("literal")

    def __lt__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda expr, other: expr < other, other=other)

    def __gt__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda expr, other: expr > other, other=other)

    def __le__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda expr, other: expr <= other, other=other)

    def __ge__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda expr, other: expr >= other, other=other)

    def __eq__(self, other: DuckDBExpr) -> Self:  # type: ignore[override]
        return self._with_callable(lambda expr, other: expr == other, other=other)

    def __ne__(self, other: DuckDBExpr) -> Self:  # type: ignore[override]
        return self._with_callable(lambda expr, other: expr != other, other=other)

    def __invert__(self) -> Self:
        invert = cast("Callable[..., Expression]", operator.invert)
        return self._with_callable(invert)

    def abs(self) -> Self:
        return self._with_callable(lambda expr: FunctionExpression("abs", expr))

    def mean(self) -> Self:
        return self._with_callable(lambda expr: FunctionExpression("mean", expr))

    def skew(self) -> Self:
        def func(expr: Expression) -> Expression:
            count = FunctionExpression("count", expr)
            # Adjust population skewness by correction factor to get sample skewness
            sample_skewness = (
                FunctionExpression("skewness", expr)
                * (count - lit(2))
                / FunctionExpression("sqrt", count * (count - lit(1)))
            )
            return when(count == lit(0), lit(None)).otherwise(
                when(count == lit(1), lit(float("nan"))).otherwise(
                    when(count == lit(2), lit(0.0)).otherwise(sample_skewness)
                )
            )

        return self._with_callable(func)

    def median(self) -> Self:
        return self._with_callable(lambda expr: FunctionExpression("median", expr))

    def all(self) -> Self:
        return self._with_callable(lambda expr: FunctionExpression("bool_and", expr))

    def any(self) -> Self:
        return self._with_callable(lambda expr: FunctionExpression("bool_or", expr))

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod
    ) -> Self:
        def func(expr: Expression) -> Expression:
            if interpolation == "linear":
                return FunctionExpression("quantile_cont", expr, lit(quantile))
            msg = "Only linear interpolation methods are supported for DuckDB quantile."
            raise NotImplementedError(msg)

        return self._with_callable(func)

    def clip(
        self,
        lower_bound: Self | NumericLiteral | TemporalLiteral | None,
        upper_bound: Self | NumericLiteral | TemporalLiteral | None,
    ) -> Self:
        def _clip_lower(expr: Expression, lower_bound: Any) -> Expression:
            return FunctionExpression("greatest", expr, lower_bound)

        def _clip_upper(expr: Expression, upper_bound: Any) -> Expression:
            return FunctionExpression("least", expr, upper_bound)

        def _clip_both(
            expr: Expression, lower_bound: Any, upper_bound: Any
        ) -> Expression:
            return FunctionExpression(
                "greatest", FunctionExpression("least", expr, upper_bound), lower_bound
            )

        if lower_bound is None:
            return self._with_callable(_clip_upper, upper_bound=upper_bound)
        if upper_bound is None:
            return self._with_callable(_clip_lower, lower_bound=lower_bound)
        return self._with_callable(
            _clip_both, lower_bound=lower_bound, upper_bound=upper_bound
        )

    def sum(self) -> Self:
        return self._with_callable(lambda expr: FunctionExpression("sum", expr))

    def n_unique(self) -> Self:
        def func(expr: Expression) -> Expression:
            # https://stackoverflow.com/a/79338887/4451315
            return FunctionExpression(
                "array_unique", FunctionExpression("array_agg", expr)
            ) + FunctionExpression(
                "max", when(expr.isnotnull(), lit(0)).otherwise(lit(1))
            )

        return self._with_callable(func)

    def count(self) -> Self:
        return self._with_callable(lambda expr: FunctionExpression("count", expr))

    def len(self) -> Self:
        return self._with_callable(lambda _expr: FunctionExpression("count"))

    def std(self, ddof: int) -> Self:
        if ddof == 0:
            return self._with_callable(
                lambda expr: FunctionExpression("stddev_pop", expr)
            )
        if ddof == 1:
            return self._with_callable(
                lambda expr: FunctionExpression("stddev_samp", expr)
            )

        def _std(expr: Expression) -> Expression:
            n_samples = FunctionExpression("count", expr)
            return (
                FunctionExpression("stddev_pop", expr)
                * FunctionExpression("sqrt", n_samples)
                / (FunctionExpression("sqrt", (n_samples - lit(ddof))))
            )

        return self._with_callable(_std)

    def var(self, ddof: int) -> Self:
        if ddof == 0:
            return self._with_callable(lambda expr: FunctionExpression("var_pop", expr))
        if ddof == 1:
            return self._with_callable(lambda expr: FunctionExpression("var_samp", expr))

        def _var(expr: Expression) -> Expression:
            n_samples = FunctionExpression("count", expr)
            return (
                FunctionExpression("var_pop", expr) * n_samples / (n_samples - lit(ddof))
            )

        return self._with_callable(_var)

    def max(self) -> Self:
        return self._with_callable(lambda expr: FunctionExpression("max", expr))

    def min(self) -> Self:
        return self._with_callable(lambda expr: FunctionExpression("min", expr))

    def null_count(self) -> Self:
        return self._with_callable(
            lambda expr: FunctionExpression("sum", expr.isnull().cast("int"))
        )

    @requires.backend_version((1, 3))
    def over(self, partition_by: Sequence[str], order_by: Sequence[str]) -> Self:
        def func(df: DuckDBLazyFrame) -> Sequence[Expression]:
            return self._window_function(df, WindowInputs(partition_by, order_by))

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def is_null(self) -> Self:
        return self._with_callable(lambda expr: expr.isnull())

    def is_nan(self) -> Self:
        return self._with_callable(lambda expr: FunctionExpression("isnan", expr))

    def is_finite(self) -> Self:
        return self._with_callable(lambda expr: FunctionExpression("isfinite", expr))

    def is_in(self, other: Sequence[Any]) -> Self:
        return self._with_callable(
            lambda expr: FunctionExpression("contains", lit(other), expr)
        )

    def round(self, decimals: int) -> Self:
        return self._with_callable(
            lambda expr: FunctionExpression("round", expr, lit(decimals))
        )

    @requires.backend_version((1, 3))
    def shift(self, n: int) -> Self:
        def func(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> Sequence[Expression]:
            order_by_sql = generate_order_by_sql(*inputs.order_by, ascending=True)
            partition_by_sql = generate_partition_by_sql(*inputs.partition_by)
            sql = f"lag({{expr}}, {n}) over ({partition_by_sql} {order_by_sql})"
            return [SQLExpression(sql.format(expr=expr)) for expr in self(df)]

        return self._with_window_function(func)

    @requires.backend_version((1, 3))
    def is_first_distinct(self) -> Self:
        def func(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> Sequence[Expression]:
            order_by_sql = generate_order_by_sql(*inputs.order_by, ascending=True)
            if inputs.partition_by:
                partition_by_sql = (
                    generate_partition_by_sql(*inputs.partition_by) + ", {expr}"
                )
            else:
                partition_by_sql = "partition by {expr}"
            sql = (
                f"{FunctionExpression('row_number')} "
                f"over({partition_by_sql} {order_by_sql})"
            )
            return [SQLExpression(sql.format(expr=expr)) == lit(1) for expr in self(df)]

        return self._with_window_function(func)

    @requires.backend_version((1, 3))
    def is_last_distinct(self) -> Self:
        def func(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> Sequence[Expression]:
            order_by_sql = generate_order_by_sql(*inputs.order_by, ascending=False)
            if inputs.partition_by:
                partition_by_sql = (
                    generate_partition_by_sql(*inputs.partition_by) + ", {expr}"
                )
            else:
                partition_by_sql = "partition by {expr}"
            sql = (
                f"{FunctionExpression('row_number')} "
                f"over({partition_by_sql} {order_by_sql})"
            )
            return [SQLExpression(sql.format(expr=expr)) == lit(1) for expr in self(df)]

        return self._with_window_function(func)

    @requires.backend_version((1, 3))
    def diff(self) -> Self:
        def func(df: DuckDBLazyFrame, inputs: DuckDBWindowInputs) -> list[Expression]:
            order_by_sql = generate_order_by_sql(*inputs.order_by, ascending=True)
            partition_by_sql = generate_partition_by_sql(*inputs.partition_by)
            sql = f"lag({{expr}}) over ({partition_by_sql} {order_by_sql})"
            return [expr - SQLExpression(sql.format(expr=expr)) for expr in self(df)]

        return self._with_window_function(func)

    @requires.backend_version((1, 3))
    def cum_sum(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="sum")
        )

    @requires.backend_version((1, 3))
    def cum_max(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="max")
        )

    @requires.backend_version((1, 3))
    def cum_min(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="min")
        )

    @requires.backend_version((1, 3))
    def cum_count(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="count")
        )

    @requires.backend_version((1, 3))
    def cum_prod(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="product")
        )

    @requires.backend_version((1, 3))
    def rolling_sum(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                func_name="sum",
                center=center,
                window_size=window_size,
                min_samples=min_samples,
            )
        )

    @requires.backend_version((1, 3))
    def rolling_mean(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                func_name="mean",
                center=center,
                window_size=window_size,
                min_samples=min_samples,
            )
        )

    @requires.backend_version((1, 3))
    def rolling_var(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                func_name="var",
                center=center,
                window_size=window_size,
                min_samples=min_samples,
                ddof=ddof,
            )
        )

    @requires.backend_version((1, 3))
    def rolling_std(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                func_name="std",
                center=center,
                window_size=window_size,
                min_samples=min_samples,
                ddof=ddof,
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
                order_by_sql = generate_order_by_sql(*inputs.order_by, ascending=True)
                partition_by_sql = generate_partition_by_sql(*inputs.partition_by)

                fill_func = "last_value" if strategy == "forward" else "first_value"
                _limit = "unbounded" if limit is None else limit
                rows_between = (
                    f"{_limit} preceding and current row"
                    if strategy == "forward"
                    else f"current row and {_limit} following"
                )
                sql = (
                    f"{fill_func}({{expr}} ignore nulls) over "
                    f"({partition_by_sql} {order_by_sql} rows between {rows_between})"
                )
                return [SQLExpression(sql.format(expr=expr)) for expr in self(df)]

            return self._with_window_function(_fill_with_strategy)

        def _fill_constant(expr: Expression, value: Any) -> Expression:
            return CoalesceOperator(expr, value)

        return self._with_callable(_fill_constant, value=value)

    def cast(self, dtype: DType | type[DType]) -> Self:
        def func(expr: Expression) -> Expression:
            native_dtype = narwhals_to_native_dtype(dtype, self._version)
            return expr.cast(DuckDBPyType(native_dtype))

        return self._with_callable(func)

    @requires.backend_version((1, 3))
    def is_unique(self) -> Self:
        def func(expr: Expression) -> Expression:
            sql = f"count(*) over (partition by {expr})"
            return SQLExpression(sql) == lit(1)

        return self._with_callable(func)

    @requires.backend_version((1, 3))
    def rank(self, method: RankMethod, *, descending: bool) -> Self:
        if method in {"min", "max", "average"}:
            func = FunctionExpression("rank")
        elif method == "dense":
            func = FunctionExpression("dense_rank")
        else:  # method == "ordinal"
            func = FunctionExpression("row_number")

        def _rank(
            expr: Expression,
            *,
            descending: bool,
            partition_by: Sequence[str | Expression] | None = None,
        ) -> Expression:
            order_by_sql = (
                f"order by {expr} desc nulls last"
                if descending
                else f"order by {expr} asc nulls last"
            )
            count_expr = FunctionExpression("count", StarExpression())
            if partition_by is not None:
                window = f"{generate_partition_by_sql(*partition_by)} {order_by_sql}"
                count_window = f"{generate_partition_by_sql(*partition_by, expr)}"
            else:
                window = order_by_sql
                count_window = generate_partition_by_sql(expr)
            if method == "max":
                rank_expr = (
                    SQLExpression(f"{func} OVER ({window})")
                    + SQLExpression(f"{count_expr} over ({count_window})")
                    - lit(1)
                )
            elif method == "average":
                rank_expr = SQLExpression(f"{func} OVER ({window})") + (
                    SQLExpression(f"{count_expr} over ({count_window})") - lit(1)
                ) / lit(2.0)
            else:
                rank_expr = SQLExpression(f"{func} OVER ({window})")
            return when(expr.isnotnull(), rank_expr)

        def _unpartitioned_rank(expr: Expression) -> Expression:
            return _rank(expr, descending=descending)

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
            log = FunctionExpression("log", expr)
            return (
                when(expr < lit(0), lit(float("nan")))
                .when(expr == lit(0), lit(float("-inf")))
                .otherwise(log / FunctionExpression("log", lit(base)))
            )

        return self._with_callable(_log)

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
