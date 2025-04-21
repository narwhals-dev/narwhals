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

    from narwhals._compliant.typing import AliasNames
    from narwhals._compliant.typing import EvalNames
    from narwhals._compliant.typing import EvalSeries
    from narwhals._expression_parsing import ExprMetadata
    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.namespace import SparkLikeNamespace
    from narwhals._spark_like.typing import WindowFunction
    from narwhals.dtypes import DType
    from narwhals.typing import FillNullStrategy
    from narwhals.typing import NonNestedLiteral
    from narwhals.typing import NumericLiteral
    from narwhals.typing import RankMethod
    from narwhals.typing import TemporalLiteral
    from narwhals.utils import Version
    from narwhals.utils import _FullContext


class SparkLikeExpr(LazyExpr["SparkLikeLazyFrame", "Column"]):
    def __init__(
        self,
        call: EvalSeries[SparkLikeLazyFrame, Column],
        *,
        evaluate_output_names: EvalNames[SparkLikeLazyFrame],
        alias_output_names: AliasNames | None,
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

    def __call__(self, df: SparkLikeLazyFrame) -> Sequence[Column]:
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
    def _F(self):  # type: ignore[no-untyped-def] # noqa: ANN202, N802
        if TYPE_CHECKING:
            from sqlframe.base import functions

            return functions
        else:
            return import_functions(self._implementation)

    @property
    def _native_dtypes(self):  # type: ignore[no-untyped-def] # noqa: ANN202
        if TYPE_CHECKING:
            from sqlframe.base import types

            return types
        else:
            return import_native_dtypes(self._implementation)

    @property
    def _Window(self) -> type[Window]:  # noqa: N802
        if TYPE_CHECKING:
            from sqlframe.base.window import Window

            return Window
        else:
            return import_window(self._implementation)

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> SparkLikeNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._spark_like.namespace import SparkLikeNamespace

        return SparkLikeNamespace(
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def _with_window_function(self, window_function: WindowFunction) -> Self:
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
        self,
        *,
        reverse: bool,
        func_name: Literal["sum", "max", "min", "count", "product"],
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

    def _rolling_window_func(
        self,
        *,
        func_name: Literal["sum", "mean", "std", "var"],
        center: bool,
        window_size: int,
        min_samples: int,
        ddof: int | None = None,
    ) -> WindowFunction:
        supported_funcs = ["sum", "mean", "std", "var"]
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
            return self._F.when(
                self._F.count(window_inputs.expr).over(window) >= min_samples,
                getattr(self._F, func_)(window_inputs.expr).over(window),
            )

        return func

    @classmethod
    def from_column_names(
        cls: type[Self],
        evaluate_column_names: EvalNames[SparkLikeLazyFrame],
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

    def _with_callable(
        self,
        call: Callable[..., Column],
        /,
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

    def _with_alias_output_names(self, func: AliasNames | None, /) -> Self:
        return type(self)(
            call=self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=func,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def __eq__(self, other: SparkLikeExpr) -> Self:  # type: ignore[override]
        return self._with_callable(
            lambda _input, other: _input.__eq__(other), other=other
        )

    def __ne__(self, other: SparkLikeExpr) -> Self:  # type: ignore[override]
        return self._with_callable(
            lambda _input, other: _input.__ne__(other), other=other
        )

    def __add__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__add__(other), other=other
        )

    def __sub__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__sub__(other), other=other
        )

    def __rsub__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__sub__(_input), other=other
        ).alias("literal")

    def __mul__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__mul__(other), other=other
        )

    def __truediv__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__truediv__(other), other=other
        )

    def __rtruediv__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__truediv__(_input), other=other
        ).alias("literal")

    def __floordiv__(self, other: SparkLikeExpr) -> Self:
        def _floordiv(_input: Column, other: Column) -> Column:
            return self._F.floor(_input / other)

        return self._with_callable(_floordiv, other=other)

    def __rfloordiv__(self, other: SparkLikeExpr) -> Self:
        def _rfloordiv(_input: Column, other: Column) -> Column:
            return self._F.floor(other / _input)

        return self._with_callable(_rfloordiv, other=other).alias("literal")

    def __pow__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__pow__(other), other=other
        )

    def __rpow__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__pow__(_input), other=other
        ).alias("literal")

    def __mod__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__mod__(other), other=other
        )

    def __rmod__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__mod__(_input), other=other
        ).alias("literal")

    def __ge__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__ge__(other), other=other
        )

    def __gt__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(lambda _input, other: _input > other, other=other)

    def __le__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__le__(other), other=other
        )

    def __lt__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__lt__(other), other=other
        )

    def __and__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__and__(other), other=other
        )

    def __or__(self, other: SparkLikeExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__or__(other), other=other
        )

    def __invert__(self) -> Self:
        invert = cast("Callable[..., Column]", operator.invert)
        return self._with_callable(invert)

    def abs(self) -> Self:
        return self._with_callable(self._F.abs)

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
            implementation=self._implementation,
        )

    def all(self) -> Self:
        return self._with_callable(self._F.bool_and)

    def any(self) -> Self:
        return self._with_callable(self._F.bool_or)

    def cast(self, dtype: DType | type[DType]) -> Self:
        def _cast(_input: Column) -> Column:
            spark_dtype = narwhals_to_native_dtype(
                dtype, self._version, self._native_dtypes
            )
            return _input.cast(spark_dtype)

        return self._with_callable(_cast)

    def count(self) -> Self:
        return self._with_callable(self._F.count)

    def max(self) -> Self:
        return self._with_callable(self._F.max)

    def mean(self) -> Self:
        return self._with_callable(self._F.mean)

    def median(self) -> Self:
        def _median(_input: Column) -> Column:
            if (
                self._implementation.is_pyspark()
                and (pyspark := get_pyspark()) is not None
                and parse_version(pyspark) < (3, 4)
            ):  # pragma: no cover
                # Use percentile_approx with default accuracy parameter (10000)
                return self._F.percentile_approx(_input.cast("double"), 0.5)

            return self._F.median(_input)

        return self._with_callable(_median)

    def min(self) -> Self:
        return self._with_callable(self._F.min)

    def null_count(self) -> Self:
        def _null_count(_input: Column) -> Column:
            return self._F.count_if(self._F.isnull(_input))

        return self._with_callable(_null_count)

    def sum(self) -> Self:
        return self._with_callable(self._F.sum)

    def std(self, ddof: int) -> Self:
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

        return self._with_callable(func)

    def var(self, ddof: int) -> Self:
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

        return self._with_callable(func)

    def clip(
        self,
        lower_bound: Self | NumericLiteral | TemporalLiteral | None = None,
        upper_bound: Self | NumericLiteral | TemporalLiteral | None = None,
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
            return (
                self._F.when(_input < lower_bound, lower_bound)
                .when(_input > upper_bound, upper_bound)
                .otherwise(_input)
            )

        if lower_bound is None:
            return self._with_callable(_clip_upper, upper_bound=upper_bound)
        if upper_bound is None:
            return self._with_callable(_clip_lower, lower_bound=lower_bound)
        return self._with_callable(
            _clip_both, lower_bound=lower_bound, upper_bound=upper_bound
        )

    def is_finite(self) -> Self:
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

        return self._with_callable(_is_finite)

    def is_in(self, values: Sequence[Any]) -> Self:
        def _is_in(_input: Column) -> Column:
            return _input.isin(values) if values else self._F.lit(False)  # noqa: FBT003

        return self._with_callable(_is_in)

    def is_unique(self) -> Self:
        def _is_unique(_input: Column) -> Column:
            # Create a window spec that treats each value separately
            return self._F.count("*").over(self._Window.partitionBy(_input)) == 1

        return self._with_callable(_is_unique)

    def len(self) -> Self:
        def _len(_input: Column) -> Column:
            # Use count(*) to count all rows including nulls
            return self._F.count("*")

        return self._with_callable(_len)

    def round(self, decimals: int) -> Self:
        def _round(_input: Column) -> Column:
            return self._F.round(_input, decimals)

        return self._with_callable(_round)

    def skew(self) -> Self:
        return self._with_callable(self._F.skewness)

    def n_unique(self) -> Self:
        def _n_unique(_input: Column) -> Column:
            return self._F.count_distinct(_input) + self._F.max(
                self._F.isnull(_input).cast(self._native_dtypes.IntegerType())
            )

        return self._with_callable(_n_unique)

    def over(
        self,
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

    def is_null(self) -> Self:
        return self._with_callable(self._F.isnull)

    def is_nan(self) -> Self:
        def _is_nan(_input: Column) -> Column:
            return self._F.when(self._F.isnull(_input), None).otherwise(
                self._F.isnan(_input)
            )

        return self._with_callable(_is_nan)

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

    def cum_count(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="count")
        )

    def cum_prod(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="product")
        )

    def fill_null(
        self,
        value: Self | NonNestedLiteral,
        strategy: FillNullStrategy | None,
        limit: int | None,
    ) -> Self:
        if strategy is not None:
            msg = "Support for strategies is not yet implemented."
            raise NotImplementedError(msg)

        def _fill_null(_input: Column, value: Column) -> Column:
            return self._F.ifnull(_input, value)

        return self._with_callable(_fill_null, value=value)

    def rolling_sum(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                func_name="sum",
                center=center,
                window_size=window_size,
                min_samples=min_samples,
            )
        )

    def rolling_mean(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                func_name="mean",
                center=center,
                window_size=window_size,
                min_samples=min_samples,
            )
        )

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

    def rank(self, method: RankMethod, *, descending: bool) -> Self:
        if method in {"min", "max", "average"}:
            func_name = "rank"
        elif method == "dense":
            func_name = "dense_rank"
        else:  # method == "ordinal"
            func_name = "row_number"

        def _rank(_input: Column) -> Column:
            if descending:
                order_by_cols = [self._F.desc_nulls_last(_input)]
            else:
                order_by_cols = [self._F.asc_nulls_last(_input)]

            window = self._Window().orderBy(order_by_cols)
            count_window = self._Window().partitionBy(_input)

            if method == "max":
                expr = (
                    getattr(self._F, func_name)().over(window)
                    + self._F.count(_input).over(count_window)
                    - self._F.lit(1)
                )

            elif method == "average":
                expr = getattr(self._F, func_name)().over(window) + (
                    self._F.count(_input).over(count_window) - self._F.lit(1)
                ) / self._F.lit(2)

            else:
                expr = getattr(self._F, func_name)().over(window)

            return self._F.when(_input.isNotNull(), expr)

        return self._with_callable(_rank)

    @property
    def str(self) -> SparkLikeExprStringNamespace:
        return SparkLikeExprStringNamespace(self)

    @property
    def dt(self) -> SparkLikeExprDateTimeNamespace:
        return SparkLikeExprDateTimeNamespace(self)

    @property
    def list(self) -> SparkLikeExprListNamespace:
        return SparkLikeExprListNamespace(self)

    @property
    def struct(self) -> SparkLikeExprStructNamespace:
        return SparkLikeExprStructNamespace(self)

    drop_nulls = not_implemented()
    unique = not_implemented()
    quantile = not_implemented()
