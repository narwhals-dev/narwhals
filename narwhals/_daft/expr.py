from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, cast

from daft import Window, coalesce, col, lit
from daft.functions import row_number

from narwhals._compliant import LazyExpr
from narwhals._compliant.window import WindowInputs
from narwhals._daft.expr_dt import DaftExprDateTimeNamespace
from narwhals._daft.expr_str import DaftExprStringNamespace
from narwhals._daft.expr_struct import DaftExprStructNamespace
from narwhals._daft.utils import maybe_evaluate_expr, narwhals_to_native_dtype
from narwhals._expression_parsing import ExprKind
from narwhals._utils import Implementation, not_implemented

if TYPE_CHECKING:
    from daft import Expression
    from typing_extensions import Self

    from narwhals._compliant.typing import AliasNames, EvalNames, WindowFunction
    from narwhals._daft.dataframe import DaftLazyFrame
    from narwhals._daft.namespace import DaftNamespace
    from narwhals._expression_parsing import ExprMetadata
    from narwhals._utils import Version, _FullContext
    from narwhals.dtypes import DType

    DaftWindowFunction = WindowFunction[DaftLazyFrame, Expression]
    DaftWindowInputs = WindowInputs[Expression]


class DaftExpr(LazyExpr["DaftLazyFrame", "Expression"]):
    _implementation = Implementation.DAFT

    def __init__(
        self,
        call: Callable[[DaftLazyFrame], Sequence[Expression]],
        window_function: DaftWindowFunction | None = None,
        *,
        evaluate_output_names: EvalNames[DaftLazyFrame],
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
        self._window_function: DaftWindowFunction | None = window_function

    @property
    def window_function(self) -> DaftWindowFunction:
        def default_window_func(
            df: DaftLazyFrame, window_inputs: DaftWindowInputs
        ) -> list[Expression]:
            assert not window_inputs.order_by  # noqa: S101
            return [
                expr.over(self.partition_by(*window_inputs.partition_by))
                for expr in self(df)
            ]

        return self._window_function or default_window_func

    def __call__(self, df: DaftLazyFrame) -> Sequence[Expression]:
        return self._call(df)

    def broadcast(self, kind: Literal[ExprKind.AGGREGATION, ExprKind.LITERAL]) -> Self:
        if kind is ExprKind.LITERAL:
            return self
        return self.over([lit(1)], [])

    def partition_by(self, *cols: Expression | str) -> Window:
        """Wraps `Window().paritionBy`, with default and `WindowInputs` handling."""
        return Window().partition_by(*cols or [lit(1)])

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> DaftNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._daft.namespace import DaftNamespace

        return DaftNamespace(backend_version=self._backend_version, version=self._version)

    def _with_window_function(self, window_function: DaftWindowFunction) -> Self:
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

    def _cum_window_func(
        self,
        *,
        reverse: bool,
        func_name: Literal["sum", "max", "min", "count", "product"],
    ) -> DaftWindowFunction:
        def func(df: DaftLazyFrame, inputs: DaftWindowInputs) -> Sequence[Expression]:
            window = (
                self.partition_by(*inputs.partition_by)
                .order_by(*inputs.order_by, desc=reverse)
                .rows_between(Window.unbounded_preceding, 0)
            )
            return [getattr(expr, func_name)().over(window) for expr in self._call(df)]

        return func

    @classmethod
    def from_column_names(
        cls: type[Self],
        evaluate_column_names: EvalNames[DaftLazyFrame],
        /,
        *,
        context: _FullContext,
    ) -> Self:
        def func(df: DaftLazyFrame) -> list[Expression]:
            return [col(col_name) for col_name in evaluate_column_names(df)]

        return cls(
            func,
            evaluate_output_names=evaluate_column_names,
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
        )

    @classmethod
    def from_column_indices(
        cls: type[Self], *column_indices: int, context: _FullContext
    ) -> Self:
        def func(df: DaftLazyFrame) -> list[Expression]:
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
        self, call: Callable[..., Expression], /, **expressifiable_args: Self | Any
    ) -> Self:
        def func(df: DaftLazyFrame) -> list[Expression]:
            native_series_list = self._call(df)
            other_native_series = {
                key: maybe_evaluate_expr(df, value)
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

    def __eq__(self, other: DaftExpr) -> Self:  # type: ignore[override]
        return self._with_callable(lambda _input, other: _input == other, other=other)

    def __ne__(self, other: DaftExpr) -> Self:  # type: ignore[override]
        return self._with_callable(lambda _input, other: _input != other, other=other)

    def __add__(self, other: DaftExpr) -> Self:
        return self._with_callable(lambda _input, other: _input + other, other=other)

    def __sub__(self, other: DaftExpr) -> Self:
        return self._with_callable(lambda _input, other: _input - other, other=other)

    def __rsub__(self, other: DaftExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__sub__(_input), other=other
        ).alias("literal")

    def __mul__(self, other: DaftExpr) -> Self:
        return self._with_callable(lambda _input, other: _input * other, other=other)

    def __truediv__(self, other: DaftExpr) -> Self:
        return self._with_callable(lambda _input, other: _input / other, other=other)

    def __rtruediv__(self, other: DaftExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__truediv__(_input), other=other
        ).alias("literal")

    def __floordiv__(self, other: DaftExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__floordiv__(other), other=other
        )

    def __rfloordiv__(self, other: DaftExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__floordiv__(_input), other=other
        ).alias("literal")

    def __pow__(self, other: DaftExpr) -> Self:
        return self._with_callable(lambda _input, other: _input**other, other=other)

    def __rpow__(self, other: DaftExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__pow__(_input), other=other
        ).alias("literal")

    def __mod__(self, other: DaftExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__mod__(other), other=other
        )

    def __rmod__(self, other: DaftExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__mod__(_input), other=other
        ).alias("literal")

    def __ge__(self, other: DaftExpr) -> Self:
        return self._with_callable(lambda _input, other: _input >= other, other=other)

    def __le__(self, other: DaftExpr) -> Self:
        return self._with_callable(lambda _input, other: _input <= other, other=other)

    def __lt__(self, other: DaftExpr) -> Self:
        return self._with_callable(lambda _input, other: _input < other, other=other)

    def __gt__(self, other: DaftExpr) -> Self:
        return self._with_callable(lambda _input, other: _input > other, other=other)

    def __and__(self, other: DaftExpr) -> Self:
        return self._with_callable(lambda _input, other: _input & other, other=other)

    def __or__(self, other: DaftExpr) -> Self:
        return self._with_callable(lambda _input, other: _input | other, other=other)

    def __invert__(self) -> Self:
        invert = cast("Callable[..., Expression]", operator.invert)
        return self._with_callable(invert)

    def alias(self, name: str) -> Self:
        def alias_output_names(names: Sequence[str]) -> Sequence[str]:
            if len(names) != 1:
                msg = f"Expected function with single output, found output names: {names}"
                raise ValueError(msg)
            return [name]

        return self.__class__(
            self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def all(self) -> Self:
        return self._with_callable(lambda _input: _input.bool_and())

    def any(self) -> Self:
        return self._with_callable(lambda _input: _input.bool_or())

    def cast(self, dtype: DType | type[DType]) -> Self:
        def func(_input: Expression) -> Expression:
            native_dtype = narwhals_to_native_dtype(
                dtype, self._version, self._backend_version
            )
            return _input.cast(native_dtype)

        return self._with_callable(func)

    def count(self) -> Self:
        return self._with_callable(lambda _input: _input.count("valid"))

    def abs(self) -> Self:
        return self._with_callable(lambda _input: _input.abs())

    def mean(self) -> Self:
        return self._with_callable(lambda _input: _input.mean())

    def quantile(
        self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Self:
        if interpolation != "lower":
            msg = "Only `interpolation='lower'` is supported for `quantile` for Daft."
            raise NotImplementedError(msg)
        return self._with_callable(lambda _input: _input.approx_percentiles(quantile))

    def clip(
        self, lower_bound: Any | None = None, upper_bound: Any | None = None
    ) -> Self:
        def _clip_lower(_input: Expression, lower_bound: Expression) -> Expression:
            return _input.clip(lower_bound)

        def _clip_upper(_input: Expression, upper_bound: Expression) -> Expression:
            return _input.clip(max=upper_bound)

        def _clip_both(
            _input: Expression, lower_bound: Expression, upper_bound: Expression
        ) -> Expression:
            return _input.clip(lower_bound, upper_bound)

        if lower_bound is None:
            return self._with_callable(_clip_upper, upper_bound=upper_bound)
        if upper_bound is None:
            return self._with_callable(_clip_lower, lower_bound=lower_bound)
        return self._with_callable(
            _clip_both, lower_bound=lower_bound, upper_bound=upper_bound
        )

    def sum(self) -> Self:
        def f(expr: Expression) -> Expression:
            return coalesce(expr.sum(), lit(0))

        def window_f(
            df: DaftLazyFrame, window_inputs: DaftWindowInputs
        ) -> Sequence[Expression]:
            return [
                coalesce(
                    expr.sum().over(self.partition_by(*window_inputs.partition_by)),
                    lit(0),
                )
                for expr in self(df)
            ]

        return self._with_callable(f)._with_window_function(window_f)

    def n_unique(self) -> Self:
        return self._with_callable(
            lambda _input: _input.count_distinct() + _input.is_null().bool_or()
        )

    def over(
        self, partition_by: Sequence[str | Expression], order_by: Sequence[str]
    ) -> Self:
        def func(df: DaftLazyFrame) -> Sequence[Expression]:
            return self.window_function(df, WindowInputs(partition_by, order_by))

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def len(self) -> Self:
        return self._with_callable(lambda _input: _input.count("all"))

    def std(self, ddof: int) -> Self:
        def func(expr: Expression) -> Expression:
            std_pop = expr.stddev()
            if ddof == 0:
                return std_pop
            n_samples = expr.count(mode="valid")
            return std_pop * n_samples.sqrt() / (n_samples - ddof).sqrt()

        return self._with_callable(func)

    def var(self, ddof: int) -> Self:
        def func(expr: Expression) -> Expression:
            std_pop = expr.stddev()
            var_pop = std_pop * std_pop
            if ddof == 0:
                return var_pop
            n_samples = expr.count(mode="valid")
            return var_pop * n_samples / (n_samples - ddof)

        return self._with_callable(func)

    def max(self) -> Self:
        return self._with_callable(lambda _input: _input.max())

    def min(self) -> Self:
        return self._with_callable(lambda _input: _input.min())

    def null_count(self) -> Self:
        return self._with_callable(lambda _input: _input.is_null().cast("uint32").sum())

    def is_null(self) -> Self:
        return self._with_callable(lambda _input: _input.is_null())

    def is_nan(self) -> Self:
        return self._with_callable(lambda _input: _input.float.is_nan())

    def shift(self, n: int) -> Self:
        def func(df: DaftLazyFrame, inputs: DaftWindowInputs) -> Sequence[Expression]:
            window = self.partition_by(*inputs.partition_by).order_by(*inputs.order_by)
            return [expr.lag(n).over(window) for expr in self(df)]

        return self._with_window_function(func)

    def is_first_distinct(self) -> Self:
        def func(df: DaftLazyFrame, inputs: DaftWindowInputs) -> Sequence[Expression]:
            return [
                row_number().over(
                    self.partition_by(*inputs.partition_by, expr).order_by(
                        *inputs.order_by
                    )
                )
                == lit(1)
                for expr in self(df)
            ]

        return self._with_window_function(func)

    def is_last_distinct(self) -> Self:
        def func(df: DaftLazyFrame, inputs: DaftWindowInputs) -> Sequence[Expression]:
            return [
                row_number().over(
                    self.partition_by(*inputs.partition_by, expr).order_by(
                        *inputs.order_by, desc=True
                    )
                )
                == lit(1)
                for expr in self(df)
            ]

        return self._with_window_function(func)

    def diff(self) -> Self:
        def func(df: DaftLazyFrame, inputs: DaftWindowInputs) -> Sequence[Expression]:
            window = self.partition_by(*inputs.partition_by).order_by(*inputs.order_by)
            return [expr - expr.lag(1).over(window) for expr in self(df)]

        return self._with_window_function(func)

    def cum_sum(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="sum")
        )

    def is_finite(self) -> Self:
        return self._with_callable(
            lambda _input: (_input > float("-inf")) & (_input < float("inf"))
        )

    def is_in(self, other: Sequence[Any]) -> Self:
        return self._with_callable(lambda _input: _input.is_in(other))

    def round(self, decimals: int) -> Self:
        return self._with_callable(lambda _input: _input.round(decimals))

    def fill_null(self, value: Self | Any, strategy: Any, limit: int | None) -> Self:
        if strategy is not None:
            msg = "todo"
            raise NotImplementedError(msg)

        return self._with_callable(
            lambda _input: _input.fill_null(value, strategy=strategy)
        )

    def log(self, base: float) -> Self:
        return self._with_callable(lambda expr: expr.log(base=base))

    def skew(self) -> Self:
        return self._with_callable(lambda expr: expr.skew())

    @property
    def str(self) -> DaftExprStringNamespace:
        return DaftExprStringNamespace(self)

    @property
    def dt(self) -> DaftExprDateTimeNamespace:
        return DaftExprDateTimeNamespace(self)

    @property
    def list(self) -> Any:
        msg = "todo"
        raise NotImplementedError(msg)

    @property
    def struct(self) -> DaftExprStructNamespace:
        return DaftExprStructNamespace(self)

    drop_nulls = not_implemented()
    rank = not_implemented()  # https://github.com/Eventual-Inc/Daft/issues/4290
    median = not_implemented()  # https://github.com/Eventual-Inc/Daft/issues/3491
    unique = not_implemented()
    is_unique = not_implemented()
    cum_count = not_implemented()
    cum_min = not_implemented()
    cum_max = not_implemented()
    cum_prod = not_implemented()
    rolling_sum = not_implemented()
    rolling_mean = not_implemented()
    rolling_var = not_implemented()
    rolling_std = not_implemented()
