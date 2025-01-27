from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence

from pyspark.sql import Window
from pyspark.sql import functions as F  # noqa: N812

from narwhals._spark_like.expr_dt import SparkLikeExprDateTimeNamespace
from narwhals._spark_like.expr_name import SparkLikeExprNameNamespace
from narwhals._spark_like.expr_str import SparkLikeExprStringNamespace
from narwhals._spark_like.utils import binary_operation_returns_scalar
from narwhals._spark_like.utils import maybe_evaluate
from narwhals._spark_like.utils import narwhals_to_native_dtype
from narwhals.typing import CompliantExpr
from narwhals.utils import Implementation
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from pyspark.sql import Column
    from typing_extensions import Self

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.namespace import SparkLikeNamespace
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class SparkLikeExpr(CompliantExpr["Column"]):
    _implementation = Implementation.PYSPARK

    def __init__(
        self: Self,
        call: Callable[[SparkLikeLazyFrame], list[Column]],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[SparkLikeLazyFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        # Whether the expression is a length-1 Column resulting from
        # a reduction, such as `nw.col('a').sum()`
        returns_scalar: bool,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._returns_scalar = returns_scalar
        self._backend_version = backend_version
        self._version = version

    def __call__(self: Self, df: SparkLikeLazyFrame) -> Sequence[Column]:
        return self._call(df)

    def __narwhals_expr__(self: Self) -> None: ...

    def __narwhals_namespace__(self: Self) -> SparkLikeNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._spark_like.namespace import SparkLikeNamespace

        return SparkLikeNamespace(
            backend_version=self._backend_version, version=self._version
        )

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(_: SparkLikeLazyFrame) -> list[Column]:
            return [F.col(col_name) for col_name in column_names]

        return cls(
            func,
            depth=0,
            function_name="col",
            evaluate_output_names=lambda _df: list(column_names),
            alias_output_names=None,
            returns_scalar=False,
            backend_version=backend_version,
            version=version,
        )

    @classmethod
    def from_column_indices(
        cls: type[Self],
        *column_indices: int,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            columns = df.columns
            return [F.col(columns[i]) for i in column_indices]

        return cls(
            func,
            depth=0,
            function_name="nth",
            evaluate_output_names=lambda df: [df.columns[i] for i in column_indices],
            alias_output_names=None,
            returns_scalar=False,
            backend_version=backend_version,
            version=version,
        )

    def _from_call(
        self: Self,
        call: Callable[..., Column],
        expr_name: str,
        *,
        returns_scalar: bool,
        **expressifiable_args: Self | Any,
    ) -> Self:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            native_series_list = self._call(df)
            other_native_series = {
                key: maybe_evaluate(df, value, returns_scalar=returns_scalar)
                for key, value in expressifiable_args.items()
            }
            return [
                call(native_series, **other_native_series)
                for native_series in native_series_list
            ]

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=f"{self._function_name}->{expr_name}",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            returns_scalar=returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
        )

    def __eq__(self: Self, other: SparkLikeExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input.__eq__(other),
            "__eq__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __ne__(self: Self, other: SparkLikeExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input.__ne__(other),
            "__ne__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __add__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__add__(other),
            "__add__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __sub__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__sub__(other),
            "__sub__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __mul__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mul__(other),
            "__mul__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __truediv__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__truediv__(other),
            "__truediv__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __floordiv__(self: Self, other: SparkLikeExpr) -> Self:
        def _floordiv(_input: Column, other: Column) -> Column:
            return F.floor(_input / other)

        return self._from_call(
            _floordiv,
            "__floordiv__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __pow__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__pow__(other),
            "__pow__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __mod__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mod__(other),
            "__mod__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __ge__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__ge__(other),
            "__ge__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __gt__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input > other,
            "__gt__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __le__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__le__(other),
            "__le__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __lt__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__lt__(other),
            "__lt__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __and__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__and__(other),
            "__and__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __or__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__or__(other),
            "__or__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __invert__(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.__invert__(),
            "__invert__",
            returns_scalar=self._returns_scalar,
        )

    def abs(self: Self) -> Self:
        return self._from_call(F.abs, "abs", returns_scalar=self._returns_scalar)

    def alias(self: Self, name: str) -> Self:
        def alias_output_names(names: Sequence[str]) -> Sequence[str]:
            if len(names) != 1:
                msg = f"Expected function with single output, found output names: {names}"
                raise ValueError(msg)
            return [name]

        # Define this one manually, so that we can
        # override `output_names` and not increase depth
        return self.__class__(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=alias_output_names,
            returns_scalar=self._returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
        )

    def all(self: Self) -> Self:
        return self._from_call(F.bool_and, "all", returns_scalar=True)

    def any(self: Self) -> Self:
        return self._from_call(F.bool_or, "any", returns_scalar=True)

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        def _cast(_input: Column) -> Column:
            spark_dtype = narwhals_to_native_dtype(dtype, self._version)
            return _input.cast(spark_dtype)

        return self._from_call(_cast, "cast", returns_scalar=self._returns_scalar)

    def count(self: Self) -> Self:
        return self._from_call(F.count, "count", returns_scalar=True)

    def max(self: Self) -> Self:
        return self._from_call(F.max, "max", returns_scalar=True)

    def mean(self: Self) -> Self:
        return self._from_call(F.mean, "mean", returns_scalar=True)

    def median(self: Self) -> Self:
        def _median(_input: Column) -> Column:
            import pyspark  # ignore-banned-import

            if parse_version(pyspark.__version__) < (3, 4):
                # Use percentile_approx with default accuracy parameter (10000)
                return F.percentile_approx(_input.cast("double"), 0.5)

            return F.median(_input)

        return self._from_call(_median, "median", returns_scalar=True)

    def min(self: Self) -> Self:
        return self._from_call(F.min, "min", returns_scalar=True)

    def null_count(self: Self) -> Self:
        def _null_count(_input: Column) -> Column:
            return F.count_if(F.isnull(_input))

        return self._from_call(_null_count, "null_count", returns_scalar=True)

    def sum(self: Self) -> Self:
        return self._from_call(F.sum, "sum", returns_scalar=True)

    def std(self: Self, ddof: int) -> Self:
        from functools import partial

        import numpy as np  # ignore-banned-import

        from narwhals._spark_like.utils import _std

        func = partial(_std, ddof=ddof, np_version=parse_version(np.__version__))

        return self._from_call(func, "std", returns_scalar=True)

    def var(self: Self, ddof: int) -> Self:
        from functools import partial

        import numpy as np  # ignore-banned-import

        from narwhals._spark_like.utils import _var

        func = partial(_var, ddof=ddof, np_version=parse_version(np.__version__))

        return self._from_call(func, "var", returns_scalar=True)

    def clip(
        self: Self,
        lower_bound: Any | None = None,
        upper_bound: Any | None = None,
    ) -> Self:
        def _clip(_input: Column, lower_bound: Any, upper_bound: Any) -> Column:
            result = _input
            if lower_bound is not None:
                # Convert lower_bound to a literal Column
                result = F.when(result < lower_bound, F.lit(lower_bound)).otherwise(
                    result
                )
            if upper_bound is not None:
                # Convert upper_bound to a literal Column
                result = F.when(result > upper_bound, F.lit(upper_bound)).otherwise(
                    result
                )
            return result

        return self._from_call(
            _clip,
            "clip",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            returns_scalar=self._returns_scalar,
        )

    def is_between(
        self: Self,
        lower_bound: Any,
        upper_bound: Any,
        closed: Literal["left", "right", "none", "both"],
    ) -> Self:
        def _is_between(_input: Column, lower_bound: Any, upper_bound: Any) -> Column:
            if closed == "both":
                return (_input >= lower_bound) & (_input <= upper_bound)
            if closed == "none":
                return (_input > lower_bound) & (_input < upper_bound)
            if closed == "left":
                return (_input >= lower_bound) & (_input < upper_bound)
            return (_input > lower_bound) & (_input <= upper_bound)

        return self._from_call(
            _is_between,
            "is_between",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            returns_scalar=self._returns_scalar,
        )

    def is_duplicated(self: Self) -> Self:
        def _is_duplicated(_input: Column) -> Column:
            # Create a window spec that treats each value separately.
            return F.count("*").over(Window.partitionBy(_input)) > 1

        return self._from_call(
            _is_duplicated, "is_duplicated", returns_scalar=self._returns_scalar
        )

    def is_finite(self: Self) -> Self:
        def _is_finite(_input: Column) -> Column:
            # A value is finite if it's not NaN, and not infinite, while NULLs should be
            # preserved
            is_finite_condition = (
                ~F.isnan(_input) & (_input != float("inf")) & (_input != float("-inf"))
            )
            return F.when(~F.isnull(_input), is_finite_condition).otherwise(None)

        return self._from_call(
            _is_finite, "is_finite", returns_scalar=self._returns_scalar
        )

    def is_in(self: Self, values: Sequence[Any]) -> Self:
        def _is_in(_input: Column) -> Column:
            return _input.isin(values)

        return self._from_call(
            _is_in,
            "is_in",
            returns_scalar=self._returns_scalar,
        )

    def is_unique(self: Self) -> Self:
        def _is_unique(_input: Column) -> Column:
            # Create a window spec that treats each value separately
            return F.count("*").over(Window.partitionBy(_input)) == 1

        return self._from_call(
            _is_unique, "is_unique", returns_scalar=self._returns_scalar
        )

    def len(self: Self) -> Self:
        def _len(_input: Column) -> Column:
            # Use count(*) to count all rows including nulls
            return F.count("*")

        return self._from_call(_len, "len", returns_scalar=True)

    def round(self: Self, decimals: int) -> Self:
        def _round(_input: Column) -> Column:
            return F.round(_input, decimals)

        return self._from_call(
            _round,
            "round",
            returns_scalar=self._returns_scalar,
        )

    def skew(self: Self) -> Self:
        return self._from_call(F.skewness, "skew", returns_scalar=True)

    def n_unique(self: Self) -> Self:
        from pyspark.sql.types import IntegerType

        def _n_unique(_input: Column) -> Column:
            return F.count_distinct(_input) + F.max(F.isnull(_input).cast(IntegerType()))

        return self._from_call(_n_unique, "n_unique", returns_scalar=True)

    def over(self: Self, keys: list[str]) -> Self:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            return [expr.over(Window.partitionBy(*keys)) for expr in self._call(df)]

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=self._function_name + "->over",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            returns_scalar=False,
        )

    def is_null(self: Self) -> Self:
        return self._from_call(F.isnull, "is_null", returns_scalar=self._returns_scalar)

    def is_nan(self: Self) -> Self:
        def _is_nan(_input: Column) -> Column:
            return F.when(F.isnull(_input), None).otherwise(F.isnan(_input))

        return self._from_call(_is_nan, "is_nan", returns_scalar=self._returns_scalar)

    @property
    def str(self: Self) -> SparkLikeExprStringNamespace:
        return SparkLikeExprStringNamespace(self)

    @property
    def name(self: Self) -> SparkLikeExprNameNamespace:
        return SparkLikeExprNameNamespace(self)

    @property
    def dt(self: Self) -> SparkLikeExprDateTimeNamespace:
        return SparkLikeExprDateTimeNamespace(self)
