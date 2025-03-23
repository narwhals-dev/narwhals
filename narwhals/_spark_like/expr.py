from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence
from typing import cast

from narwhals._compliant import LazyExpr
from narwhals._expression_parsing import ExprKind
from narwhals._spark_like.expr_dt import SparkLikeExprDateTimeNamespace
from narwhals._spark_like.expr_list import SparkLikeExprListNamespace
from narwhals._spark_like.expr_name import SparkLikeExprNameNamespace
from narwhals._spark_like.expr_str import SparkLikeExprStringNamespace
from narwhals._spark_like.expr_struct import SparkLikeExprStructNamespace
from narwhals._spark_like.utils import WindowInputs
from narwhals._spark_like.utils import import_functions
from narwhals._spark_like.utils import import_native_dtypes
from narwhals._spark_like.utils import import_window
from narwhals._spark_like.utils import narwhals_to_native_dtype
from narwhals.dependencies import get_pyspark
from narwhals.utils import Implementation
from narwhals.utils import not_implemented
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from sqlframe.base.column import Column
    from sqlframe.base.window import Window
    from typing_extensions import Self

    from narwhals._expression_parsing import ExprMetadata
    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.namespace import SparkLikeNamespace
    from narwhals._spark_like.typing import WindowFunction
    from narwhals.dtypes import DType
    from narwhals.utils import Version
    from narwhals.utils import _FullContext


class SparkLikeExpr(LazyExpr["SparkLikeLazyFrame", "Column"]):
    def __init__(
        self: Self,
        call: Callable[[SparkLikeLazyFrame], Sequence[Column]],
        *,
        evaluate_output_names: Callable[[SparkLikeLazyFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        backend_version: tuple[int, ...],
        version: Version,
        implementation: Implementation,
    ) -> None:
        self._call = call
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._backend_version = backend_version
        self._version = version
        self._implementation = implementation
        self._window_function: WindowFunction | None = None
        self._metadata: ExprMetadata | None = None

    def __call__(self: Self, df: SparkLikeLazyFrame) -> Sequence[Column]:
        return self._call(df)

    def broadcast(self, kind: Literal[ExprKind.AGGREGATION, ExprKind.LITERAL]) -> Self:
        if kind is ExprKind.LITERAL:
            return self

        def func(df: SparkLikeLazyFrame) -> Sequence[Column]:
            return [
                result.over(df._Window().partitionBy(df._F.lit(1))) for result in self(df)
            ]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    @property
    def _F(self: Self):  # type: ignore[no-untyped-def] # noqa: ANN202, N802
        if TYPE_CHECKING:
            from sqlframe.base import functions

            return functions
        else:
            return import_functions(self._implementation)

    @property
    def _native_dtypes(self: Self):  # type: ignore[no-untyped-def] # noqa: ANN202
        if TYPE_CHECKING:
            from sqlframe.base import types

            return types
        else:
            return import_native_dtypes(self._implementation)

    @property
    def _Window(self: Self) -> type[Window]:  # noqa: N802
        if TYPE_CHECKING:
            from sqlframe.base.window import Window

            return Window
        else:
            return import_window(self._implementation)

    def __narwhals_expr__(self: Self) -> None: ...

    def __narwhals_namespace__(self: Self) -> SparkLikeNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._spark_like.namespace import SparkLikeNamespace

        return SparkLikeNamespace(
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def _with_metadata(self, metadata: ExprMetadata) -> Self:
        expr = self.__class__(
            self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )
        if func := self._window_function:
            expr = expr._with_window_function(func)
        expr._metadata = metadata
        return expr

    def _with_window_function(
        self: Self,
        window_function: WindowFunction,
    ) -> Self:
        result = self.__class__(
            self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )
        result._window_function = window_function
        return result

    def _cum_window_func(
        self: Self, *, reverse: bool, func_name: Literal["sum", "max", "min"]
    ) -> WindowFunction:
        def func(window_inputs: WindowInputs) -> Column:
            if reverse:
                order_by_cols = [
                    self._F.col(x).desc_nulls_last() for x in window_inputs.order_by
                ]
            else:
                order_by_cols = [
                    self._F.col(x).asc_nulls_first() for x in window_inputs.order_by
                ]
            window = (
                self._Window()
                .partitionBy(list(window_inputs.partition_by))
                .orderBy(order_by_cols)
                .rowsBetween(self._Window().unboundedPreceding, 0)
            )
            return getattr(self._F, func_name)(window_inputs.expr).over(window)

        return func

    @classmethod
    def from_column_names(
        cls: type[Self],
        evaluate_column_names: Callable[[SparkLikeLazyFrame], Sequence[str]],
        /,
        *,
        context: _FullContext,
    ) -> Self:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            return [df._F.col(col_name) for col_name in evaluate_column_names(df)]

        return cls(
            func,
            evaluate_output_names=evaluate_column_names,
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
            implementation=context._implementation,
        )

    @classmethod
    def from_column_indices(
        cls: type[Self], *column_indices: int, context: _FullContext
    ) -> Self:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            columns = df.columns
            return [df._F.col(columns[i]) for i in column_indices]

        return cls(
            func,
            evaluate_output_names=lambda df: [df.columns[i] for i in column_indices],
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
            implementation=context._implementation,
        )

    def _from_call(
        self: Self,
        call: Callable[..., Column],
        **expressifiable_args: Self | Any,
    ) -> Self:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            native_series_list = self(df)
            lit = df._F.lit
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
            implementation=self._implementation,
        )

    def __eq__(self: Self, other: SparkLikeExpr) -> Self:  # type: ignore[override]
        return self._from_call(lambda _input, other: _input.__eq__(other), other=other)

    def __ne__(self: Self, other: SparkLikeExpr) -> Self:  # type: ignore[override]
        return self._from_call(lambda _input, other: _input.__ne__(other), other=other)

    def __add__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(lambda _input, other: _input.__add__(other), other=other)

    def __sub__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(lambda _input, other: _input.__sub__(other), other=other)

    def __rsub__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: other.__sub__(_input), other=other
        ).alias("literal")

    def __mul__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(lambda _input, other: _input.__mul__(other), other=other)

    def __truediv__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__truediv__(other), other=other
        )

    def __rtruediv__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: other.__truediv__(_input), other=other
        ).alias("literal")

    def __floordiv__(self: Self, other: SparkLikeExpr) -> Self:
        def _floordiv(_input: Column, other: Column) -> Column:
            return self._F.floor(_input / other)

        return self._from_call(_floordiv, other=other)

    def __rfloordiv__(self: Self, other: SparkLikeExpr) -> Self:
        def _rfloordiv(_input: Column, other: Column) -> Column:
            return self._F.floor(other / _input)

        return self._from_call(_rfloordiv, other=other).alias("literal")

    def __pow__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(lambda _input, other: _input.__pow__(other), other=other)

    def __rpow__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: other.__pow__(_input), other=other
        ).alias("literal")

    def __mod__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(lambda _input, other: _input.__mod__(other), other=other)

    def __rmod__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: other.__mod__(_input), other=other
        ).alias("literal")

    def __ge__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(lambda _input, other: _input.__ge__(other), other=other)

    def __gt__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(lambda _input, other: _input > other, other=other)

    def __le__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(lambda _input, other: _input.__le__(other), other=other)

    def __lt__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(lambda _input, other: _input.__lt__(other), other=other)

    def __and__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(lambda _input, other: _input.__and__(other), other=other)

    def __or__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(lambda _input, other: _input.__or__(other), other=other)

    def __invert__(self: Self) -> Self:
        invert = cast("Callable[..., Column]", operator.invert)
        return self._from_call(invert)

    def abs(self: Self) -> Self:
        return self._from_call(self._F.abs)

    def alias(self: Self, name: str) -> Self:
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
            implementation=self._implementation,
        )

    def all(self: Self) -> Self:
        return self._from_call(self._F.bool_and)

    def any(self: Self) -> Self:
        return self._from_call(self._F.bool_or)

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        def _cast(_input: Column) -> Column:
            spark_dtype = narwhals_to_native_dtype(
                dtype, self._version, self._native_dtypes
            )
            return _input.cast(spark_dtype)

        return self._from_call(_cast)

    def count(self: Self) -> Self:
        return self._from_call(self._F.count)

    def max(self: Self) -> Self:
        return self._from_call(self._F.max)

    def mean(self: Self) -> Self:
        return self._from_call(self._F.mean)

    def median(self: Self) -> Self:
        def _median(_input: Column) -> Column:
            if (
                self._implementation.is_pyspark()
                and (pyspark := get_pyspark()) is not None
                and parse_version(pyspark) < (3, 4)
            ):  # pragma: no cover
                # Use percentile_approx with default accuracy parameter (10000)
                return self._F.percentile_approx(_input.cast("double"), 0.5)

            return self._F.median(_input)

        return self._from_call(_median)

    def min(self: Self) -> Self:
        return self._from_call(self._F.min)

    def null_count(self: Self) -> Self:
        def _null_count(_input: Column) -> Column:
            return self._F.count_if(self._F.isnull(_input))

        return self._from_call(_null_count)

    def sum(self: Self) -> Self:
        return self._from_call(self._F.sum)

    def std(self: Self, ddof: int) -> Self:
        from functools import partial

        import numpy as np  # ignore-banned-import

        from narwhals._spark_like.utils import _std

        func = partial(
            _std,
            ddof=ddof,
            np_version=parse_version(np),
            functions=self._F,
            implementation=self._implementation,
        )

        return self._from_call(func)

    def var(self: Self, ddof: int) -> Self:
        from functools import partial

        import numpy as np  # ignore-banned-import

        from narwhals._spark_like.utils import _var

        func = partial(
            _var,
            ddof=ddof,
            np_version=parse_version(np),
            functions=self._F,
            implementation=self._implementation,
        )

        return self._from_call(func)

    def clip(
        self: Self,
        lower_bound: Any | None = None,
        upper_bound: Any | None = None,
    ) -> Self:
        def _clip_lower(_input: Column, lower_bound: Column) -> Column:
            result = _input
            return self._F.when(result < lower_bound, lower_bound).otherwise(result)

        def _clip_upper(_input: Column, upper_bound: Column) -> Column:
            result = _input
            return self._F.when(result > upper_bound, upper_bound).otherwise(result)

        def _clip_both(
            _input: Column, lower_bound: Column, upper_bound: Column
        ) -> Column:
            result = _input
            result = self._F.when(result < lower_bound, lower_bound).otherwise(result)
            return self._F.when(result > upper_bound, upper_bound).otherwise(result)

        if lower_bound is None:
            return self._from_call(_clip_upper, upper_bound=upper_bound)
        if upper_bound is None:
            return self._from_call(_clip_lower, lower_bound=lower_bound)
        return self._from_call(
            _clip_both, lower_bound=lower_bound, upper_bound=upper_bound
        )

    def is_finite(self: Self) -> Self:
        def _is_finite(_input: Column) -> Column:
            # A value is finite if it's not NaN, and not infinite, while NULLs should be
            # preserved
            is_finite_condition = (
                ~self._F.isnan(_input)
                & (_input != self._F.lit(float("inf")))
                & (_input != self._F.lit(float("-inf")))
            )
            return self._F.when(~self._F.isnull(_input), is_finite_condition).otherwise(
                None
            )

        return self._from_call(_is_finite)

    def is_in(self: Self, values: Sequence[Any]) -> Self:
        def _is_in(_input: Column) -> Column:
            return _input.isin(values) if values else self._F.lit(False)  # noqa: FBT003

        return self._from_call(_is_in)

    def is_unique(self: Self) -> Self:
        def _is_unique(_input: Column) -> Column:
            # Create a window spec that treats each value separately
            return self._F.count("*").over(self._Window.partitionBy(_input)) == 1

        return self._from_call(_is_unique)

    def len(self: Self) -> Self:
        def _len(_input: Column) -> Column:
            # Use count(*) to count all rows including nulls
            return self._F.count("*")

        return self._from_call(_len)

    def round(self: Self, decimals: int) -> Self:
        def _round(_input: Column) -> Column:
            return self._F.round(_input, decimals)

        return self._from_call(_round)

    def skew(self: Self) -> Self:
        return self._from_call(self._F.skewness)

    def n_unique(self: Self) -> Self:
        def _n_unique(_input: Column) -> Column:
            return self._F.count_distinct(_input) + self._F.max(
                self._F.isnull(_input).cast(self._native_dtypes.IntegerType())
            )

        return self._from_call(_n_unique)

    def over(
        self: Self,
        partition_by: Sequence[str],
        order_by: Sequence[str] | None,
    ) -> Self:
        if (window_function := self._window_function) is not None:
            assert order_by is not None  # noqa: S101

            def func(df: SparkLikeLazyFrame) -> list[Column]:
                return [
                    window_function(WindowInputs(expr, partition_by, order_by))
                    for expr in self._call(df)
                ]
        else:

            def func(df: SparkLikeLazyFrame) -> list[Column]:
                return [
                    expr.over(self._Window.partitionBy(*partition_by))
                    for expr in self._call(df)
                ]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def is_null(self: Self) -> Self:
        return self._from_call(self._F.isnull)

    def is_nan(self: Self) -> Self:
        def _is_nan(_input: Column) -> Column:
            return self._F.when(self._F.isnull(_input), None).otherwise(
                self._F.isnan(_input)
            )

        return self._from_call(_is_nan)

    def shift(self, n: int) -> Self:
        def func(window_inputs: WindowInputs) -> Column:
            order_by_cols = [
                self._F.col(x).asc_nulls_first() for x in window_inputs.order_by
            ]
            window = (
                self._Window()
                .partitionBy(list(window_inputs.partition_by))
                .orderBy(order_by_cols)
            )
            return self._F.lag(window_inputs.expr, n).over(window)

        return self._with_window_function(func)

    def is_first_distinct(self) -> Self:
        def func(window_inputs: WindowInputs) -> Column:
            order_by_cols = [
                self._F.col(x).asc_nulls_first() for x in window_inputs.order_by
            ]
            window = (
                self._Window()
                .partitionBy([*window_inputs.partition_by, window_inputs.expr])
                .orderBy(order_by_cols)
            )
            return self._F.row_number().over(window) == 1

        return self._with_window_function(func)

    def is_last_distinct(self) -> Self:
        def func(window_inputs: WindowInputs) -> Column:
            order_by_cols = [
                self._F.col(x).desc_nulls_last() for x in window_inputs.order_by
            ]
            window = (
                self._Window()
                .partitionBy([*window_inputs.partition_by, window_inputs.expr])
                .orderBy(order_by_cols)
            )
            return self._F.row_number().over(window) == 1

        return self._with_window_function(func)

    def diff(self) -> Self:
        def func(window_inputs: WindowInputs) -> Column:
            order_by_cols = [
                self._F.col(x).asc_nulls_first() for x in window_inputs.order_by
            ]
            window = (
                self._Window()
                .partitionBy(list(window_inputs.partition_by))
                .orderBy(order_by_cols)
            )
            return window_inputs.expr - self._F.lag(window_inputs.expr).over(window)

        return self._with_window_function(func)

    def cum_sum(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="sum")
        )

    def cum_max(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="max")
        )

    def cum_min(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="min")
        )

    def fill_null(
        self,
        value: Any | None,
        strategy: Literal["forward", "backward"] | None,
        limit: int | None,
    ) -> Self:
        if strategy is not None:
            msg = "Support for strategies is not yet implemented."
            raise NotImplementedError(msg)

        def _fill_null(_input: Column, value: Column) -> Column:
            return self._F.ifnull(_input, value)

        return self._from_call(_fill_null, value=value)

    def rolling_sum(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        if center:
            half = (window_size - 1) // 2
            remainder = (window_size - 1) % 2
            start = self._Window().currentRow - half - remainder
            end = self._Window().currentRow + half
        else:
            start = self._Window().currentRow - window_size + 1
            end = self._Window().currentRow

        def func(window_inputs: WindowInputs) -> Column:
            window = (
                self._Window()
                .partitionBy(list(window_inputs.partition_by))
                .orderBy(
                    [self._F.col(x).asc_nulls_first() for x in window_inputs.order_by]
                )
                .rowsBetween(start, end)
            )
            return self._F.when(
                self._F.count(window_inputs.expr).over(window) >= min_samples,
                self._F.sum(window_inputs.expr).over(window),
            )

        return self._with_window_function(func)

    @property
    def str(self: Self) -> SparkLikeExprStringNamespace:
        return SparkLikeExprStringNamespace(self)

    @property
    def name(self: Self) -> SparkLikeExprNameNamespace:
        return SparkLikeExprNameNamespace(self)

    @property
    def dt(self: Self) -> SparkLikeExprDateTimeNamespace:
        return SparkLikeExprDateTimeNamespace(self)

    @property
    def list(self: Self) -> SparkLikeExprListNamespace:
        return SparkLikeExprListNamespace(self)

    @property
    def struct(self: Self) -> SparkLikeExprStructNamespace:
        return SparkLikeExprStructNamespace(self)

    drop_nulls = not_implemented()
    unique = not_implemented()
    cum_count = not_implemented()
    cum_prod = not_implemented()
    quantile = not_implemented()
