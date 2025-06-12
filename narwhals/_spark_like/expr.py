from __future__ import annotations

import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    cast,
)

from narwhals._compliant import LazyExpr
from narwhals._compliant.window import WindowInputs
from narwhals._expression_parsing import ExprKind
from narwhals._spark_like.expr_dt import SparkLikeExprDateTimeNamespace
from narwhals._spark_like.expr_list import SparkLikeExprListNamespace
from narwhals._spark_like.expr_str import SparkLikeExprStringNamespace
from narwhals._spark_like.expr_struct import SparkLikeExprStructNamespace
from narwhals._spark_like.utils import (
    import_functions,
    import_native_dtypes,
    import_window,
    narwhals_to_native_dtype,
)
from narwhals._utils import Implementation, not_implemented, parse_version
from narwhals.dependencies import get_pyspark

if TYPE_CHECKING:
    from sqlframe.base.column import Column
    from sqlframe.base.window import Window, WindowSpec
    from typing_extensions import Self, TypeAlias

    from narwhals._compliant.typing import (
        AliasNames,
        EvalNames,
        EvalSeries,
        WindowFunction,
    )
    from narwhals._expression_parsing import ExprMetadata
    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.namespace import SparkLikeNamespace
    from narwhals._utils import Version, _FullContext
    from narwhals.dtypes import DType
    from narwhals.typing import (
        FillNullStrategy,
        NonNestedLiteral,
        NumericLiteral,
        RankMethod,
        TemporalLiteral,
    )

    NativeRankMethod: TypeAlias = Literal["rank", "dense_rank", "row_number"]
    Asc: TypeAlias = Literal[False]
    Desc: TypeAlias = Literal[True]
    NullsFirst: TypeAlias = Literal[False]
    NullsLast: TypeAlias = Literal[True]

    SparkWindowFunction = WindowFunction[SparkLikeLazyFrame, Column]
    SparkWindowInputs = WindowInputs[Column]

ASC_NULLS_FIRST: tuple[Asc, NullsFirst] = False, False
ASC_NULLS_LAST: tuple[Asc, NullsLast] = False, True
DESC_NULLS_FIRST: tuple[Desc, NullsFirst] = True, False
DESC_NULLS_LAST: tuple[Desc, NullsLast] = True, True


class SparkLikeExpr(LazyExpr["SparkLikeLazyFrame", "Column"]):
    _REMAP_RANK_METHOD: ClassVar[Mapping[RankMethod, NativeRankMethod]] = {
        "min": "rank",
        "max": "rank",
        "average": "rank",
        "dense": "dense_rank",
        "ordinal": "row_number",
    }

    def __init__(
        self,
        call: EvalSeries[SparkLikeLazyFrame, Column],
        window_function: SparkWindowFunction | None = None,
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
        self._metadata: ExprMetadata | None = None
        self._window_function: SparkWindowFunction | None = window_function

    @property
    def window_function(self) -> SparkWindowFunction:
        def default_window_func(
            df: SparkLikeLazyFrame, window_inputs: SparkWindowInputs
        ) -> list[Column]:
            assert not window_inputs.order_by  # noqa: S101
            return [
                expr.over(self.partition_by(*window_inputs.partition_by))
                for expr in self(df)
            ]

        return self._window_function or default_window_func

    def __call__(self, df: SparkLikeLazyFrame) -> Sequence[Column]:
        return self._call(df)

    def broadcast(self, kind: Literal[ExprKind.AGGREGATION, ExprKind.LITERAL]) -> Self:
        if kind is ExprKind.LITERAL:
            return self
        return self.over([self._F.lit(1)], [])

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

    def _sort(
        self, *cols: Column | str, descending: bool = False, nulls_last: bool = False
    ) -> Iterator[Column]:
        """Sort one or more columns.

        Arguments:
            *cols: One or more columns, or a `WindowInputs` object - where `order_by` will be used.
            descending: Sort in descending order.
            nulls_last: Place null values last.

        Yields:
            Column expressions, in order of appearance in `cols`.
        """
        F = self._F  # noqa: N806
        mapping = {
            ASC_NULLS_FIRST: F.asc_nulls_first,
            DESC_NULLS_FIRST: F.desc_nulls_first,
            ASC_NULLS_LAST: F.asc_nulls_last,
            DESC_NULLS_LAST: F.desc_nulls_last,
        }
        sort = mapping[(descending, nulls_last)]
        yield from (sort(col) for col in cols)

    def partition_by(self, *cols: Column | str) -> WindowSpec:
        """Wraps `Window().paritionBy`, with default and `WindowInputs` handling."""
        return self._Window.partitionBy(*cols or [self._F.lit(1)])

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> SparkLikeNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._spark_like.namespace import SparkLikeNamespace

        return SparkLikeNamespace(
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def _with_window_function(self, window_function: SparkWindowFunction) -> Self:
        return self.__class__(
            self._call,
            window_function,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    @classmethod
    def _alias_native(cls, expr: Column, name: str) -> Column:
        return expr.alias(name)

    def _cum_window_func(
        self,
        *,
        reverse: bool,
        func_name: Literal["sum", "max", "min", "count", "product"],
    ) -> SparkWindowFunction:
        def func(df: SparkLikeLazyFrame, inputs: SparkWindowInputs) -> Sequence[Column]:
            window = (
                self.partition_by(*inputs.partition_by)
                .orderBy(
                    *self._sort(*inputs.order_by, descending=reverse, nulls_last=reverse)
                )
                .rowsBetween(self._Window.unboundedPreceding, 0)
            )
            return [
                getattr(self._F, func_name)(expr).over(window) for expr in self._call(df)
            ]

        return func

    def _rolling_window_func(
        self,
        *,
        func_name: Literal["sum", "mean", "std", "var"],
        center: bool,
        window_size: int,
        min_samples: int,
        ddof: int | None = None,
    ) -> SparkWindowFunction:
        supported_funcs = ["sum", "mean", "std", "var"]
        if center:
            half = (window_size - 1) // 2
            remainder = (window_size - 1) % 2
            start = self._Window.currentRow - half - remainder
            end = self._Window.currentRow + half
        else:
            start = self._Window.currentRow - window_size + 1
            end = self._Window.currentRow

        def func(df: SparkLikeLazyFrame, inputs: SparkWindowInputs) -> Sequence[Column]:
            window = (
                self.partition_by(*inputs.partition_by)
                .orderBy(*self._sort(*inputs.order_by))
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
            return [
                self._F.when(
                    self._F.count(expr).over(window) >= min_samples,
                    getattr(self._F, func_)(expr).over(window),
                )
                for expr in self._call(df)
            ]

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
    def from_column_indices(cls, *column_indices: int, context: _FullContext) -> Self:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            columns = df.columns
            return [df._F.col(columns[i]) for i in column_indices]

        return cls(
            func,
            evaluate_output_names=cls._eval_names_indices(column_indices),
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
            implementation=context._implementation,
        )

    def _callable_to_eval_series(
        self, call: Callable[..., Column], /, **expressifiable_args: Self | Any
    ) -> EvalSeries[SparkLikeLazyFrame, Column]:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            native_series_list = self(df)
            other_native_series = {
                key: df._evaluate_expr(value)
                if self._is_expr(value)
                else self._F.lit(value)
                for key, value in expressifiable_args.items()
            }
            return [
                call(native_series, **other_native_series)
                for native_series in native_series_list
            ]

        return func

    def _push_down_window_function(
        self, call: Callable[..., Column], /, **expressifiable_args: Self | Any
    ) -> SparkWindowFunction:
        def window_f(
            df: SparkLikeLazyFrame, window_inputs: SparkWindowInputs
        ) -> Sequence[Column]:
            # If a function `f` is elementwise, and `g` is another function, then
            # - `f(g) over (window)`
            # - `f(g over (window))
            # are equivalent.
            # Make sure to only use with if `call` is elementwise!
            native_series_list = self.window_function(df, window_inputs)
            other_native_series = {
                key: df._evaluate_window_expr(value, window_inputs)
                if self._is_expr(value)
                else self._F.lit(value)
                for key, value in expressifiable_args.items()
            }
            return [
                call(native_series, **other_native_series)
                for native_series in native_series_list
            ]

        return window_f

    def _with_callable(
        self, call: Callable[..., Column], /, **expressifiable_args: Self | Any
    ) -> Self:
        return self.__class__(
            self._callable_to_eval_series(call, **expressifiable_args),
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def _with_elementwise(
        self, call: Callable[..., Column], /, **expressifiable_args: Self | Any
    ) -> Self:
        return self.__class__(
            self._callable_to_eval_series(call, **expressifiable_args),
            self._push_down_window_function(call, **expressifiable_args),
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def _with_binary(self, op: Callable[..., Column], other: Self | Any) -> Self:
        return self.__class__(
            self._callable_to_eval_series(op, other=other),
            self._push_down_window_function(op, other=other),
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def _with_alias_output_names(self, func: AliasNames | None, /) -> Self:
        return type(self)(
            self._call,
            self._window_function,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=func,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def __eq__(self, other: SparkLikeExpr) -> Self:  # type: ignore[override]
        return self._with_binary(lambda expr, other: expr.__eq__(other), other)

    def __ne__(self, other: SparkLikeExpr) -> Self:  # type: ignore[override]
        return self._with_binary(lambda expr, other: expr.__ne__(other), other)

    def __add__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: expr.__add__(other), other)

    def __sub__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: expr.__sub__(other), other)

    def __rsub__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: other.__sub__(expr), other).alias(
            "literal"
        )

    def __mul__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: expr.__mul__(other), other)

    def __truediv__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: expr.__truediv__(other), other)

    def __rtruediv__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(
            lambda expr, other: other.__truediv__(expr), other
        ).alias("literal")

    def __floordiv__(self, other: SparkLikeExpr) -> Self:
        def _floordiv(expr: Column, other: Column) -> Column:
            return self._F.floor(expr / other)

        return self._with_binary(_floordiv, other)

    def __rfloordiv__(self, other: SparkLikeExpr) -> Self:
        def _rfloordiv(expr: Column, other: Column) -> Column:
            return self._F.floor(other / expr)

        return self._with_binary(_rfloordiv, other).alias("literal")

    def __pow__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: expr.__pow__(other), other)

    def __rpow__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: other.__pow__(expr), other).alias(
            "literal"
        )

    def __mod__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: expr.__mod__(other), other)

    def __rmod__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: other.__mod__(expr), other).alias(
            "literal"
        )

    def __ge__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: expr.__ge__(other), other)

    def __gt__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: expr > other, other)

    def __le__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: expr.__le__(other), other)

    def __lt__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: expr.__lt__(other), other)

    def __and__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: expr.__and__(other), other)

    def __or__(self, other: SparkLikeExpr) -> Self:
        return self._with_binary(lambda expr, other: expr.__or__(other), other)

    def __invert__(self) -> Self:
        invert = cast("Callable[..., Column]", operator.invert)
        return self._with_elementwise(invert)

    def abs(self) -> Self:
        return self._with_elementwise(self._F.abs)

    def all(self) -> Self:
        def f(expr: Column) -> Column:
            return self._F.coalesce(self._F.bool_and(expr), self._F.lit(True))  # noqa: FBT003

        def window_f(
            df: SparkLikeLazyFrame, window_inputs: SparkWindowInputs
        ) -> Sequence[Column]:
            return [
                self._F.coalesce(
                    self._F.bool_and(expr).over(
                        self.partition_by(*window_inputs.partition_by)
                    ),
                    self._F.lit(True),  # noqa: FBT003
                )
                for expr in self(df)
            ]

        return self._with_callable(f)._with_window_function(window_f)

    def any(self) -> Self:
        def f(expr: Column) -> Column:
            return self._F.coalesce(self._F.bool_or(expr), self._F.lit(False))  # noqa: FBT003

        def window_f(
            df: SparkLikeLazyFrame, window_inputs: SparkWindowInputs
        ) -> Sequence[Column]:
            return [
                self._F.coalesce(
                    self._F.bool_or(expr).over(
                        self.partition_by(*window_inputs.partition_by)
                    ),
                    self._F.lit(False),  # noqa: FBT003
                )
                for expr in self(df)
            ]

        return self._with_callable(f)._with_window_function(window_f)

    def cast(self, dtype: DType | type[DType]) -> Self:
        def _cast(expr: Column) -> Column:
            spark_dtype = narwhals_to_native_dtype(
                dtype, self._version, self._native_dtypes
            )
            return expr.cast(spark_dtype)

        return self._with_elementwise(_cast)

    def count(self) -> Self:
        return self._with_callable(self._F.count)

    def max(self) -> Self:
        return self._with_callable(self._F.max)

    def mean(self) -> Self:
        return self._with_callable(self._F.mean)

    def median(self) -> Self:
        def _median(expr: Column) -> Column:
            if (
                self._implementation
                in {Implementation.PYSPARK, Implementation.PYSPARK_CONNECT}
                and (pyspark := get_pyspark()) is not None
                and parse_version(pyspark) < (3, 4)
            ):  # pragma: no cover
                # Use percentile_approx with default accuracy parameter (10000)
                return self._F.percentile_approx(expr.cast("double"), 0.5)

            return self._F.median(expr)

        return self._with_callable(_median)

    def min(self) -> Self:
        return self._with_callable(self._F.min)

    def null_count(self) -> Self:
        def _null_count(expr: Column) -> Column:
            return self._F.count_if(self._F.isnull(expr))

        return self._with_callable(_null_count)

    def sum(self) -> Self:
        def f(expr: Column) -> Column:
            return self._F.coalesce(self._F.sum(expr), self._F.lit(0))

        def window_f(
            df: SparkLikeLazyFrame, window_inputs: SparkWindowInputs
        ) -> Sequence[Column]:
            return [
                self._F.coalesce(
                    self._F.sum(expr).over(
                        self.partition_by(*window_inputs.partition_by)
                    ),
                    self._F.lit(0),
                )
                for expr in self(df)
            ]

        return self._with_callable(f)._with_window_function(window_f)

    def std(self, ddof: int) -> Self:
        F = self._F  # noqa: N806
        if ddof == 0:
            return self._with_callable(F.stddev_pop)
        if ddof == 1:
            return self._with_callable(F.stddev_samp)

        def func(expr: Column) -> Column:
            n_rows = F.count(expr)
            return F.stddev_samp(expr) * F.sqrt((n_rows - 1) / (n_rows - ddof))

        return self._with_callable(func)

    def var(self, ddof: int) -> Self:
        F = self._F  # noqa: N806
        if ddof == 0:
            return self._with_callable(F.var_pop)
        if ddof == 1:
            return self._with_callable(F.var_samp)

        def func(expr: Column) -> Column:
            n_rows = F.count(expr)
            return F.var_samp(expr) * (n_rows - 1) / (n_rows - ddof)

        return self._with_callable(func)

    def clip(
        self,
        lower_bound: Self | NumericLiteral | TemporalLiteral | None = None,
        upper_bound: Self | NumericLiteral | TemporalLiteral | None = None,
    ) -> Self:
        def _clip_lower(expr: Column, lower_bound: Column) -> Column:
            result = expr
            return self._F.when(result < lower_bound, lower_bound).otherwise(result)

        def _clip_upper(expr: Column, upper_bound: Column) -> Column:
            result = expr
            return self._F.when(result > upper_bound, upper_bound).otherwise(result)

        def _clip_both(expr: Column, lower_bound: Column, upper_bound: Column) -> Column:
            return (
                self._F.when(expr < lower_bound, lower_bound)
                .when(expr > upper_bound, upper_bound)
                .otherwise(expr)
            )

        if lower_bound is None:
            return self._with_elementwise(_clip_upper, upper_bound=upper_bound)
        if upper_bound is None:
            return self._with_elementwise(_clip_lower, lower_bound=lower_bound)
        return self._with_elementwise(
            _clip_both, lower_bound=lower_bound, upper_bound=upper_bound
        )

    def is_finite(self) -> Self:
        def _is_finite(expr: Column) -> Column:
            # A value is finite if it's not NaN, and not infinite, while NULLs should be
            # preserved
            is_finite_condition = (
                ~self._F.isnan(expr)
                & (expr != self._F.lit(float("inf")))
                & (expr != self._F.lit(float("-inf")))
            )
            return self._F.when(~self._F.isnull(expr), is_finite_condition).otherwise(
                None
            )

        return self._with_elementwise(_is_finite)

    def is_in(self, values: Sequence[Any]) -> Self:
        def _is_in(expr: Column) -> Column:
            return expr.isin(values) if values else self._F.lit(False)  # noqa: FBT003

        return self._with_elementwise(_is_in)

    def is_unique(self) -> Self:
        def _is_unique(expr: Column, *partition_by: str | Column) -> Column:
            return self._F.count("*").over(self.partition_by(expr, *partition_by)) == 1

        def _unpartitioned_is_unique(expr: Column) -> Column:
            return _is_unique(expr)

        def _partitioned_is_unique(
            df: SparkLikeLazyFrame, inputs: SparkWindowInputs
        ) -> Sequence[Column]:
            assert not inputs.order_by  # noqa: S101
            return [_is_unique(expr, *inputs.partition_by) for expr in self(df)]

        return self._with_callable(_unpartitioned_is_unique)._with_window_function(
            _partitioned_is_unique
        )

    def len(self) -> Self:
        def _len(_expr: Column) -> Column:
            # Use count(*) to count all rows including nulls
            return self._F.count("*")

        return self._with_callable(_len)

    def round(self, decimals: int) -> Self:
        def _round(expr: Column) -> Column:
            return self._F.round(expr, decimals)

        return self._with_elementwise(_round)

    def skew(self) -> Self:
        return self._with_callable(self._F.skewness)

    def n_unique(self) -> Self:
        def _n_unique(expr: Column) -> Column:
            return self._F.count_distinct(expr) + self._F.max(
                self._F.isnull(expr).cast(self._native_dtypes.IntegerType())
            )

        return self._with_callable(_n_unique)

    def over(self, partition_by: Sequence[str | Column], order_by: Sequence[str]) -> Self:
        def func(df: SparkLikeLazyFrame) -> Sequence[Column]:
            return self.window_function(df, WindowInputs(partition_by, order_by))

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def is_null(self) -> Self:
        return self._with_elementwise(self._F.isnull)

    def is_nan(self) -> Self:
        def _is_nan(expr: Column) -> Column:
            return self._F.when(self._F.isnull(expr), None).otherwise(self._F.isnan(expr))

        return self._with_elementwise(_is_nan)

    def shift(self, n: int) -> Self:
        def func(df: SparkLikeLazyFrame, inputs: SparkWindowInputs) -> Sequence[Column]:
            window = self.partition_by(*inputs.partition_by).orderBy(
                *self._sort(*inputs.order_by)
            )
            return [self._F.lag(expr, n).over(window) for expr in self(df)]

        return self._with_window_function(func)

    def is_first_distinct(self) -> Self:
        def func(df: SparkLikeLazyFrame, inputs: SparkWindowInputs) -> Sequence[Column]:
            return [
                self._F.row_number().over(
                    self.partition_by(*inputs.partition_by, expr).orderBy(
                        *self._sort(*inputs.order_by)
                    )
                )
                == 1
                for expr in self(df)
            ]

        return self._with_window_function(func)

    def is_last_distinct(self) -> Self:
        def func(df: SparkLikeLazyFrame, inputs: SparkWindowInputs) -> Sequence[Column]:
            return [
                self._F.row_number().over(
                    self.partition_by(*inputs.partition_by, expr).orderBy(
                        *self._sort(*inputs.order_by, descending=True)
                    )
                )
                == 1
                for expr in self(df)
            ]

        return self._with_window_function(func)

    def diff(self) -> Self:
        def func(df: SparkLikeLazyFrame, inputs: SparkWindowInputs) -> Sequence[Column]:
            window = self.partition_by(*inputs.partition_by).orderBy(
                *self._sort(*inputs.order_by)
            )
            return [expr - self._F.lag(expr).over(window) for expr in self(df)]

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

            def _fill_with_strategy(
                df: SparkLikeLazyFrame, inputs: SparkWindowInputs
            ) -> Sequence[Column]:
                fn = self._F.last_value if strategy == "forward" else self._F.first_value
                if strategy == "forward":
                    start = self._Window.unboundedPreceding if limit is None else -limit
                    end = self._Window.currentRow
                else:
                    start = self._Window.currentRow
                    end = self._Window.unboundedFollowing if limit is None else limit
                return [
                    fn(expr, ignoreNulls=True).over(
                        self.partition_by(*inputs.partition_by)
                        .orderBy(*self._sort(*inputs.order_by))
                        .rowsBetween(start, end)
                    )
                    for expr in self(df)
                ]

            return self._with_window_function(_fill_with_strategy)

        def _fill_constant(expr: Column, value: Column) -> Column:
            return self._F.ifnull(expr, value)

        return self._with_elementwise(_fill_constant, value=value)

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
        func_name = self._REMAP_RANK_METHOD[method]

        def _rank(
            expr: Column,
            *,
            descending: bool,
            partition_by: Sequence[str | Column] | None = None,
        ) -> Column:
            order_by = self._sort(expr, descending=descending, nulls_last=True)
            if partition_by is not None:
                window = self.partition_by(*partition_by).orderBy(*order_by)
                count_window = self.partition_by(*partition_by, expr)
            else:
                window = self.partition_by().orderBy(*order_by)
                count_window = self.partition_by(expr)
            if method == "max":
                rank_expr = (
                    getattr(self._F, func_name)().over(window)
                    + self._F.count(expr).over(count_window)
                    - self._F.lit(1)
                )

            elif method == "average":
                rank_expr = getattr(self._F, func_name)().over(window) + (
                    self._F.count(expr).over(count_window) - self._F.lit(1)
                ) / self._F.lit(2)

            else:
                rank_expr = getattr(self._F, func_name)().over(window)

            return self._F.when(expr.isNotNull(), rank_expr)

        def _unpartitioned_rank(expr: Column) -> Column:
            return _rank(expr, descending=descending)

        def _partitioned_rank(
            df: SparkLikeLazyFrame, inputs: SparkWindowInputs
        ) -> Sequence[Column]:
            assert not inputs.order_by  # noqa: S101
            return [
                _rank(expr, descending=descending, partition_by=inputs.partition_by)
                for expr in self(df)
            ]

        return self._with_callable(_unpartitioned_rank)._with_window_function(
            _partitioned_rank
        )

    def log(self, base: float) -> Self:
        def _log(expr: Column) -> Column:
            return (
                self._F.when(expr < 0, self._F.lit(float("nan")))
                .when(expr == 0, self._F.lit(float("-inf")))
                .otherwise(self._F.log(float(base), expr))
            )

        return self._with_elementwise(_log)

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
