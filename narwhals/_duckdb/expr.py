from __future__ import annotations

import contextlib
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence
from typing import cast

from duckdb import CoalesceOperator
from duckdb import FunctionExpression
from duckdb import StarExpression
from duckdb.typing import DuckDBPyType

from narwhals._compliant import LazyExpr
from narwhals._duckdb.expr_dt import DuckDBExprDateTimeNamespace
from narwhals._duckdb.expr_list import DuckDBExprListNamespace
from narwhals._duckdb.expr_str import DuckDBExprStringNamespace
from narwhals._duckdb.expr_struct import DuckDBExprStructNamespace
from narwhals._duckdb.utils import WindowInputs
from narwhals._duckdb.utils import col
from narwhals._duckdb.utils import ensure_type
from narwhals._duckdb.utils import generate_order_by_sql
from narwhals._duckdb.utils import generate_partition_by_sql
from narwhals._duckdb.utils import lit
from narwhals._duckdb.utils import narwhals_to_native_dtype
from narwhals._duckdb.utils import when
from narwhals._expression_parsing import ExprKind
from narwhals.utils import Implementation
from narwhals.utils import not_implemented
from narwhals.utils import requires

if TYPE_CHECKING:
    import duckdb
    from typing_extensions import Self

    from narwhals._compliant.typing import AliasNames
    from narwhals._compliant.typing import EvalNames
    from narwhals._compliant.typing import EvalSeries
    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.namespace import DuckDBNamespace
    from narwhals._duckdb.typing import WindowFunction
    from narwhals._expression_parsing import ExprMetadata
    from narwhals.dtypes import DType
    from narwhals.typing import FillNullStrategy
    from narwhals.typing import NonNestedLiteral
    from narwhals.typing import NumericLiteral
    from narwhals.typing import RankMethod
    from narwhals.typing import RollingInterpolationMethod
    from narwhals.typing import TemporalLiteral
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

with contextlib.suppress(ImportError):  # requires duckdb>=1.3.0
    from duckdb import SQLExpression  # type: ignore[attr-defined, unused-ignore]


class DuckDBExpr(LazyExpr["DuckDBLazyFrame", "duckdb.Expression"]):
    _implementation = Implementation.DUCKDB

    def __init__(
        self,
        call: EvalSeries[DuckDBLazyFrame, duckdb.Expression],
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
        self._window_function: WindowFunction | None = None
        self._metadata: ExprMetadata | None = None

    def __call__(self, df: DuckDBLazyFrame) -> Sequence[duckdb.Expression]:
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
    ) -> WindowFunction:
        def func(window_inputs: WindowInputs) -> duckdb.Expression:
            order_by_sql = generate_order_by_sql(
                *window_inputs.order_by, ascending=not reverse
            )
            partition_by_sql = generate_partition_by_sql(*window_inputs.partition_by)
            sql = (
                f"{func_name} ({window_inputs.expr}) over ({partition_by_sql} {order_by_sql} "
                "rows between unbounded preceding and current row)"
            )
            return SQLExpression(sql)  # type: ignore[no-any-return, unused-ignore]

        return func

    def _rolling_window_func(
        self,
        *,
        func_name: Literal["sum", "mean", "std", "var"],
        center: bool,
        window_size: int,
        min_samples: int,
        ddof: int | None = None,
    ) -> WindowFunction:
        ensure_type(window_size, int, type(None))
        ensure_type(min_samples, int)
        supported_funcs = ["sum", "mean", "std", "var"]
        if center:
            half = (window_size - 1) // 2
            remainder = (window_size - 1) % 2
            start = f"{half + remainder} preceding"
            end = f"{half} following"
        else:
            start = f"{window_size - 1} preceding"
            end = "current row"

        def func(window_inputs: WindowInputs) -> duckdb.Expression:
            order_by_sql = generate_order_by_sql(*window_inputs.order_by, ascending=True)
            partition_by_sql = generate_partition_by_sql(*window_inputs.partition_by)
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
            condition_sql = f"count({window_inputs.expr}) over {window} >= {min_samples}"
            condition = SQLExpression(condition_sql)
            value = SQLExpression(f"{func_}({window_inputs.expr}) over {window}")
            return when(condition, value)

        return func

    def broadcast(self, kind: Literal[ExprKind.AGGREGATION, ExprKind.LITERAL]) -> Self:
        if kind is ExprKind.LITERAL:
            return self
        if self._backend_version < (1, 3):
            msg = "At least version 1.3 of DuckDB is required for binary operations between aggregates and columns."
            raise NotImplementedError(msg)

        template = "{expr} over ()"

        def func(df: DuckDBLazyFrame) -> Sequence[duckdb.Expression]:
            return [SQLExpression(template.format(expr=expr)) for expr in self(df)]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    @classmethod
    def from_column_names(
        cls,
        evaluate_column_names: EvalNames[DuckDBLazyFrame],
        /,
        *,
        context: _FullContext,
    ) -> Self:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
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
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            columns = df.columns
            return [col(columns[i]) for i in column_indices]

        return cls(
            func,
            evaluate_output_names=lambda df: [df.columns[i] for i in column_indices],
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
        )

    def _with_callable(
        self, call: Callable[..., duckdb.Expression], /, **expressifiable_args: Self | Any
    ) -> Self:
        """Create expression from callable.

        Arguments:
            call: Callable from compliant DataFrame to native Expression
            expr_name: Expression name
            expressifiable_args: arguments pass to expression which should be parsed
                as expressions (e.g. in `nw.col('a').is_between('b', 'c')`)
        """

        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
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

    def _with_window_function(self, window_function: WindowFunction) -> Self:
        result = self.__class__(
            self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )
        result._window_function = window_function
        return result

    def __and__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda _input, other: _input & other, other=other)

    def __or__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda _input, other: _input | other, other=other)

    def __add__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda _input, other: _input + other, other=other)

    def __truediv__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda _input, other: _input / other, other=other)

    def __rtruediv__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__truediv__(_input), other=other
        ).alias("literal")

    def __floordiv__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__floordiv__(other), other=other
        )

    def __rfloordiv__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__floordiv__(_input), other=other
        ).alias("literal")

    def __mod__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__mod__(other), other=other
        )

    def __rmod__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__mod__(_input), other=other
        ).alias("literal")

    def __sub__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda _input, other: _input - other, other=other)

    def __rsub__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__sub__(_input), other=other
        ).alias("literal")

    def __mul__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda _input, other: _input * other, other=other)

    def __pow__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda _input, other: _input**other, other=other)

    def __rpow__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__pow__(_input), other=other
        ).alias("literal")

    def __lt__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda _input, other: _input < other, other=other)

    def __gt__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda _input, other: _input > other, other=other)

    def __le__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda _input, other: _input <= other, other=other)

    def __ge__(self, other: DuckDBExpr) -> Self:
        return self._with_callable(lambda _input, other: _input >= other, other=other)

    def __eq__(self, other: DuckDBExpr) -> Self:  # type: ignore[override]
        return self._with_callable(lambda _input, other: _input == other, other=other)

    def __ne__(self, other: DuckDBExpr) -> Self:  # type: ignore[override]
        return self._with_callable(lambda _input, other: _input != other, other=other)

    def __invert__(self) -> Self:
        invert = cast("Callable[..., duckdb.Expression]", operator.invert)
        return self._with_callable(invert)

    def abs(self) -> Self:
        return self._with_callable(lambda _input: FunctionExpression("abs", _input))

    def mean(self) -> Self:
        return self._with_callable(lambda _input: FunctionExpression("mean", _input))

    def skew(self) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            count = FunctionExpression("count", _input)
            # Adjust population skewness by correction factor to get sample skewness
            sample_skewness = (
                FunctionExpression("skewness", _input)
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
        return self._with_callable(lambda _input: FunctionExpression("median", _input))

    def all(self) -> Self:
        return self._with_callable(lambda _input: FunctionExpression("bool_and", _input))

    def any(self) -> Self:
        return self._with_callable(lambda _input: FunctionExpression("bool_or", _input))

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod
    ) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            if interpolation == "linear":
                return FunctionExpression("quantile_cont", _input, lit(quantile))
            msg = "Only linear interpolation methods are supported for DuckDB quantile."
            raise NotImplementedError(msg)

        return self._with_callable(func)

    def clip(
        self,
        lower_bound: Self | NumericLiteral | TemporalLiteral | None,
        upper_bound: Self | NumericLiteral | TemporalLiteral | None,
    ) -> Self:
        def _clip_lower(_input: duckdb.Expression, lower_bound: Any) -> duckdb.Expression:
            return FunctionExpression("greatest", _input, lower_bound)

        def _clip_upper(_input: duckdb.Expression, upper_bound: Any) -> duckdb.Expression:
            return FunctionExpression("least", _input, upper_bound)

        def _clip_both(
            _input: duckdb.Expression, lower_bound: Any, upper_bound: Any
        ) -> duckdb.Expression:
            return FunctionExpression(
                "greatest", FunctionExpression("least", _input, upper_bound), lower_bound
            )

        if lower_bound is None:
            return self._with_callable(_clip_upper, upper_bound=upper_bound)
        if upper_bound is None:
            return self._with_callable(_clip_lower, lower_bound=lower_bound)
        return self._with_callable(
            _clip_both, lower_bound=lower_bound, upper_bound=upper_bound
        )

    def sum(self) -> Self:
        return self._with_callable(lambda _input: FunctionExpression("sum", _input))

    def n_unique(self) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            # https://stackoverflow.com/a/79338887/4451315
            return FunctionExpression(
                "array_unique", FunctionExpression("array_agg", _input)
            ) + FunctionExpression(
                "max", when(_input.isnotnull(), lit(0)).otherwise(lit(1))
            )

        return self._with_callable(func)

    def count(self) -> Self:
        return self._with_callable(lambda _input: FunctionExpression("count", _input))

    def len(self) -> Self:
        return self._with_callable(lambda _input: FunctionExpression("count"))

    def std(self, ddof: int) -> Self:
        if ddof == 0:
            return self._with_callable(
                lambda _input: FunctionExpression("stddev_pop", _input)
            )
        if ddof == 1:
            return self._with_callable(
                lambda _input: FunctionExpression("stddev_samp", _input)
            )

        def _std(_input: duckdb.Expression) -> duckdb.Expression:
            n_samples = FunctionExpression("count", _input)
            return (
                FunctionExpression("stddev_pop", _input)
                * FunctionExpression("sqrt", n_samples)
                / (FunctionExpression("sqrt", (n_samples - lit(ddof))))
            )

        return self._with_callable(_std)

    def var(self, ddof: int) -> Self:
        if ddof == 0:
            return self._with_callable(
                lambda _input: FunctionExpression("var_pop", _input)
            )
        if ddof == 1:
            return self._with_callable(
                lambda _input: FunctionExpression("var_samp", _input)
            )

        def _var(_input: duckdb.Expression) -> duckdb.Expression:
            n_samples = FunctionExpression("count", _input)
            return (
                FunctionExpression("var_pop", _input)
                * n_samples
                / (n_samples - lit(ddof))
            )

        return self._with_callable(_var)

    def max(self) -> Self:
        return self._with_callable(lambda _input: FunctionExpression("max", _input))

    def min(self) -> Self:
        return self._with_callable(lambda _input: FunctionExpression("min", _input))

    def null_count(self) -> Self:
        return self._with_callable(
            lambda _input: FunctionExpression("sum", _input.isnull().cast("int")),
        )

    @requires.backend_version((1, 3))
    def over(self, partition_by: Sequence[str], order_by: Sequence[str] | None) -> Self:
        if (window_function := self._window_function) is not None:
            assert order_by is not None  # noqa: S101

            def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
                return [
                    window_function(WindowInputs(expr, partition_by, order_by))
                    for expr in self._call(df)
                ]
        else:
            partition_by_sql = generate_partition_by_sql(*partition_by)
            template = f"{{expr}} over ({partition_by_sql})"

            def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
                return [
                    SQLExpression(template.format(expr=expr)) for expr in self._call(df)
                ]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def is_null(self) -> Self:
        return self._with_callable(lambda _input: _input.isnull())

    def is_nan(self) -> Self:
        return self._with_callable(lambda _input: FunctionExpression("isnan", _input))

    def is_finite(self) -> Self:
        return self._with_callable(lambda _input: FunctionExpression("isfinite", _input))

    def is_in(self, other: Sequence[Any]) -> Self:
        return self._with_callable(
            lambda _input: FunctionExpression("contains", lit(other), _input)
        )

    def round(self, decimals: int) -> Self:
        return self._with_callable(
            lambda _input: FunctionExpression("round", _input, lit(decimals))
        )

    @requires.backend_version((1, 3))
    def shift(self, n: int) -> Self:
        ensure_type(n, int)

        def func(window_inputs: WindowInputs) -> duckdb.Expression:
            order_by_sql = generate_order_by_sql(*window_inputs.order_by, ascending=True)
            partition_by_sql = generate_partition_by_sql(*window_inputs.partition_by)
            sql = (
                f"lag({window_inputs.expr}, {n}) over ({partition_by_sql} {order_by_sql})"
            )
            return SQLExpression(sql)  # type: ignore[no-any-return, unused-ignore]

        return self._with_window_function(func)

    @requires.backend_version((1, 3))
    def is_first_distinct(self) -> Self:
        def func(window_inputs: WindowInputs) -> duckdb.Expression:
            order_by_sql = generate_order_by_sql(*window_inputs.order_by, ascending=True)
            if window_inputs.partition_by:
                partition_by_sql = (
                    generate_partition_by_sql(*window_inputs.partition_by)
                    + f", {window_inputs.expr}"
                )
            else:
                partition_by_sql = f"partition by {window_inputs.expr}"
            sql = f"{FunctionExpression('row_number')} over({partition_by_sql} {order_by_sql})"
            return SQLExpression(sql) == lit(1)  # type: ignore[no-any-return, unused-ignore]

        return self._with_window_function(func)

    @requires.backend_version((1, 3))
    def is_last_distinct(self) -> Self:
        def func(window_inputs: WindowInputs) -> duckdb.Expression:
            order_by_sql = generate_order_by_sql(*window_inputs.order_by, ascending=False)
            if window_inputs.partition_by:
                partition_by_sql = (
                    generate_partition_by_sql(*window_inputs.partition_by)
                    + f", {window_inputs.expr}"
                )
            else:
                partition_by_sql = f"partition by {window_inputs.expr}"
            sql = f"{FunctionExpression('row_number')} over({partition_by_sql} {order_by_sql})"
            return SQLExpression(sql) == lit(1)  # type: ignore[no-any-return, unused-ignore]

        return self._with_window_function(func)

    @requires.backend_version((1, 3))
    def diff(self) -> Self:
        def func(window_inputs: WindowInputs) -> duckdb.Expression:
            order_by_sql = generate_order_by_sql(*window_inputs.order_by, ascending=True)
            partition_by_sql = generate_partition_by_sql(*window_inputs.partition_by)
            sql = f"lag({window_inputs.expr}) over ({partition_by_sql} {order_by_sql})"
            return window_inputs.expr - SQLExpression(sql)  # type: ignore[no-any-return, unused-ignore]

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

            def _fill_with_strategy(window_inputs: WindowInputs) -> duckdb.Expression:
                order_by_sql = generate_order_by_sql(
                    *window_inputs.order_by, ascending=True
                )
                partition_by_sql = generate_partition_by_sql(*window_inputs.partition_by)

                fill_func = "last_value" if strategy == "forward" else "first_value"
                _limit = "unbounded" if limit is None else limit
                rows_between = (
                    f"{_limit} preceding and current row"
                    if strategy == "forward"
                    else f"current row and {_limit} following"
                )
                sql = (
                    f"{fill_func}({window_inputs.expr} ignore nulls) over "
                    f"({partition_by_sql} {order_by_sql} rows between {rows_between})"
                )
                return SQLExpression(sql)  # type: ignore[no-any-return, unused-ignore]

            return self._with_window_function(_fill_with_strategy)

        def _fill_constant(_input: duckdb.Expression, value: Any) -> duckdb.Expression:
            return CoalesceOperator(_input, value)

        return self._with_callable(_fill_constant, value=value)

    def cast(self, dtype: DType | type[DType]) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            native_dtype = narwhals_to_native_dtype(dtype, self._version)
            return _input.cast(DuckDBPyType(native_dtype))

        return self._with_callable(func)

    @requires.backend_version((1, 3))
    def is_unique(self) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            sql = f"count(*) over (partition by {_input})"
            return SQLExpression(sql) == lit(1)  # type: ignore[no-any-return, unused-ignore]

        return self._with_callable(func)

    @requires.backend_version((1, 3))
    def rank(self, method: RankMethod, *, descending: bool) -> Self:
        if method in {"min", "max", "average"}:
            func = FunctionExpression("rank")
        elif method == "dense":
            func = FunctionExpression("dense_rank")
        else:  # method == "ordinal"
            func = FunctionExpression("row_number")

        def _rank(_input: duckdb.Expression) -> duckdb.Expression:
            if descending:
                by_sql = f"{_input} desc nulls last"
            else:
                by_sql = f"{_input} asc nulls last"
            order_by_sql = f"order by {by_sql}"
            count_expr = FunctionExpression("count", StarExpression())

            if method == "max":
                expr = (
                    SQLExpression(f"{func} OVER ({order_by_sql})")
                    + SQLExpression(f"{count_expr} OVER (PARTITION BY {_input})")
                    - lit(1)
                )
            elif method == "average":
                expr = SQLExpression(f"{func} OVER ({order_by_sql})") + (
                    SQLExpression(f"{count_expr} OVER (PARTITION BY {_input})") - lit(1)
                ) / lit(2.0)
            else:
                expr = SQLExpression(f"{func} OVER ({order_by_sql})")

            return when(_input.isnotnull(), expr)

        return self._with_callable(_rank)

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
