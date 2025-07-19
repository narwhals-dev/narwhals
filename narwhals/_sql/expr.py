from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

from narwhals._compliant.typing import AliasNames, CompliantLazyFrameT, NativeExprT
from narwhals._typing_compat import Protocol38
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from typing_extensions import Self, TypeIs

    from narwhals._compliant.typing import AliasNames, WindowFunction
    from narwhals._compliant.window import WindowInputs

from narwhals._compliant.expr import LazyExpr
from narwhals._compliant.typing import WindowFunction


class SQLExpr(
    LazyExpr[CompliantLazyFrameT, NativeExprT],
    Protocol38[CompliantLazyFrameT, NativeExprT],
):
    _window_function: WindowFunction[CompliantLazyFrameT, NativeExprT] | None

    @property
    def window_function(self) -> WindowFunction[CompliantLazyFrameT, NativeExprT]:
        def default_window_func(
            df: CompliantLazyFrameT, inputs: WindowInputs[NativeExprT]
        ) -> Sequence[NativeExprT]:
            assert not inputs.order_by  # noqa: S101
            return [
                self._window_expression(expr, inputs.partition_by, inputs.order_by)
                for expr in self(df)
            ]

        return self._window_function or default_window_func

    def _function(self, name: str, *args: NativeExprT) -> NativeExprT: ...
    def _lit(self, value: Any) -> NativeExprT: ...
    def _window_expression(
        self,
        expr: NativeExprT,
        partition_by: Sequence[str | NativeExprT] = (),
        order_by: Sequence[str | NativeExprT] = (),
        rows_start: str = "",
        rows_end: str = "",
        *,
        descending: Sequence[bool] | None = None,
        nulls_last: Sequence[bool] | None = None,
    ) -> NativeExprT: ...

    def _cum_window_func(
        self,
        func_name: Literal["sum", "max", "min", "count", "product"],
        *,
        reverse: bool,
    ) -> WindowFunction[CompliantLazyFrameT, NativeExprT]:
        def func(
            df: CompliantLazyFrameT, inputs: WindowInputs[NativeExprT]
        ) -> Sequence[NativeExprT]:
            return [
                self._window_expression(
                    self._function(func_name, expr),
                    inputs.partition_by,
                    inputs.order_by,
                    descending=[reverse] * len(inputs.order_by),
                    nulls_last=[reverse] * len(inputs.order_by),
                    rows_start="unbounded preceding",
                    rows_end="current row",
                )
                for expr in self(df)
            ]

        return func

    @classmethod
    def _is_expr(cls, obj: Self | Any) -> TypeIs[Self]:
        return hasattr(obj, "__narwhals_expr__")

    def _with_callable(self, call: Callable[..., Any], /) -> Self: ...
    def _with_elementwise(
        self, op: Callable[..., NativeExprT], /, **expressifiable_args: Self | Any
    ) -> Self: ...
    def _with_binary(self, op: Callable[..., NativeExprT], other: Self | Any) -> Self: ...
    def _with_window_function(
        self, window_function: WindowFunction[CompliantLazyFrameT, NativeExprT]
    ) -> Self: ...

    def _with_alias_output_names(self, func: AliasNames | None, /) -> Self: ...

    @property
    def _backend_version(self) -> tuple[int, ...]:
        return self._implementation._backend_version()

    @classmethod
    def _alias_native(cls, expr: NativeExprT, name: str, /) -> NativeExprT: ...

    @classmethod
    def _from_elementwise_horizontal_op(
        cls, func: Callable[[Iterable[NativeExprT]], NativeExprT], *exprs: Self
    ) -> Self: ...

    def __eq__(self, other: Self) -> Self:  # type: ignore[override]
        return self._with_binary(lambda expr, other: expr.__eq__(other), other)

    def __ne__(self, other: Self) -> Self:  # type: ignore[override]
        return self._with_binary(lambda expr, other: expr.__ne__(other), other)

    def __add__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__add__(other), other)

    def __sub__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__sub__(other), other)

    def __rsub__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: other - expr, other).alias("literal")

    def __mul__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__mul__(other), other)

    def __truediv__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__truediv__(other), other)

    def __rtruediv__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: other / expr, other).alias("literal")

    def __floordiv__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__floordiv__(other), other)

    def __rfloordiv__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: other // expr, other).alias(
            "literal"
        )

    def __pow__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__pow__(other), other)

    def __rpow__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: other**expr, other).alias("literal")

    def __mod__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__mod__(other), other)

    def __rmod__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: other % expr, other).alias("literal")

    def __ge__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__ge__(other), other)

    def __gt__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__gt__(other), other)

    def __le__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__le__(other), other)

    def __lt__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__lt__(other), other)

    def __and__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__and__(other), other)

    def __or__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: expr.__or__(other), other)

    # Aggregations
    def max(self) -> Self:
        return self._with_callable(lambda expr: self._function("max", expr))

    def mean(self) -> Self:
        return self._with_callable(lambda expr: self._function("mean", expr))

    def median(self) -> Self:
        return self._with_callable(lambda expr: self._function("median", expr))

    def min(self) -> Self:
        return self._with_callable(lambda expr: self._function("min", expr))

    # Elementwise
    def abs(self) -> Self:
        return self._with_elementwise(lambda expr: self._function("abs", expr))

    def is_null(self) -> Self:
        return self._with_elementwise(lambda expr: self._function("isnull", expr))

    def round(self, decimals: int) -> Self:
        return self._with_elementwise(
            lambda expr: self._function("round", expr, self._lit(decimals))
        )

    # Cumulative
    def cum_sum(self, *, reverse: bool) -> Self:
        return self._with_window_function(self._cum_window_func("sum", reverse=reverse))

    def cum_max(self, *, reverse: bool) -> Self:
        return self._with_window_function(self._cum_window_func("max", reverse=reverse))

    def cum_min(self, *, reverse: bool) -> Self:
        return self._with_window_function(self._cum_window_func("min", reverse=reverse))

    def cum_count(self, *, reverse: bool) -> Self:
        return self._with_window_function(self._cum_window_func("count", reverse=reverse))

    def cum_prod(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func("product", reverse=reverse)
        )

    arg_max: not_implemented = not_implemented()
    arg_min: not_implemented = not_implemented()
    arg_true: not_implemented = not_implemented()
    ewm_mean: not_implemented = not_implemented()
    gather_every: not_implemented = not_implemented()
    head: not_implemented = not_implemented()
    map_batches: not_implemented = not_implemented()
    mode: not_implemented = not_implemented()
    replace_strict: not_implemented = not_implemented()
    sort: not_implemented = not_implemented()
    sample: not_implemented = not_implemented()
    tail: not_implemented = not_implemented()

    # namespaces
    cat: not_implemented = not_implemented()  # type: ignore[assignment]
