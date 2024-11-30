from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence

from narwhals._expression_parsing import reuse_series_implementation
from narwhals._expression_parsing import reuse_series_namespace_implementation
from narwhals._pandas_like.series import PandasLikeSeries
from narwhals.dependencies import get_numpy
from narwhals.dependencies import is_numpy_array
from narwhals.exceptions import ColumnNotFoundError

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.namespace import PandasLikeNamespace
    from narwhals.dtypes import DType
    from narwhals.typing import DTypes
    from narwhals.utils import Implementation


class PandasLikeExpr:
    def __init__(
        self,
        call: Callable[[PandasLikeDataFrame], list[PandasLikeSeries]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._depth = depth
        self._output_names = output_names
        self._implementation = implementation
        self._backend_version = backend_version
        self._dtypes = dtypes

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PandasLikeExpr("
            f"depth={self._depth}, "
            f"function_name={self._function_name}, "
            f"root_names={self._root_names}, "
            f"output_names={self._output_names}"
        )

    def __narwhals_namespace__(self) -> PandasLikeNamespace:
        from narwhals._pandas_like.namespace import PandasLikeNamespace

        return PandasLikeNamespace(
            self._implementation, self._backend_version, dtypes=self._dtypes
        )

    def __narwhals_expr__(self) -> None: ...

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> Self:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            try:
                return [
                    PandasLikeSeries(
                        df._native_frame[column_name],
                        implementation=df._implementation,
                        backend_version=df._backend_version,
                        dtypes=df._dtypes,
                    )
                    for column_name in column_names
                ]
            except KeyError as e:
                missing_columns = [x for x in column_names if x not in df.columns]
                raise ColumnNotFoundError.from_missing_and_available_column_names(
                    missing_columns=missing_columns,
                    available_columns=df.columns,
                ) from e

        return cls(
            func,
            depth=0,
            function_name="col",
            root_names=list(column_names),
            output_names=list(column_names),
            implementation=implementation,
            backend_version=backend_version,
            dtypes=dtypes,
        )

    @classmethod
    def from_column_indices(
        cls: type[Self],
        *column_indices: int,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> Self:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            return [
                PandasLikeSeries(
                    df._native_frame.iloc[:, column_index],
                    implementation=df._implementation,
                    backend_version=df._backend_version,
                    dtypes=df._dtypes,
                )
                for column_index in column_indices
            ]

        return cls(
            func,
            depth=0,
            function_name="nth",
            root_names=None,
            output_names=None,
            implementation=implementation,
            backend_version=backend_version,
            dtypes=dtypes,
        )

    def cast(
        self,
        dtype: Any,
    ) -> Self:
        return reuse_series_implementation(self, "cast", dtype=dtype)

    def __eq__(self, other: PandasLikeExpr | Any) -> Self:  # type: ignore[override]
        return reuse_series_implementation(self, "__eq__", other=other)

    def __ne__(self, other: PandasLikeExpr | Any) -> Self:  # type: ignore[override]
        return reuse_series_implementation(self, "__ne__", other=other)

    def __ge__(self, other: PandasLikeExpr | Any) -> Self:
        return reuse_series_implementation(self, "__ge__", other=other)

    def __gt__(self, other: PandasLikeExpr | Any) -> Self:
        return reuse_series_implementation(self, "__gt__", other=other)

    def __le__(self, other: PandasLikeExpr | Any) -> Self:
        return reuse_series_implementation(self, "__le__", other=other)

    def __lt__(self, other: PandasLikeExpr | Any) -> Self:
        return reuse_series_implementation(self, "__lt__", other=other)

    def __and__(self, other: PandasLikeExpr | bool | Any) -> Self:
        return reuse_series_implementation(self, "__and__", other=other)

    def __rand__(self, other: Any) -> Self:
        other = self.__narwhals_namespace__().lit(other, dtype=None)
        return other.__and__(self)  # type: ignore[no-any-return]

    def __or__(self, other: PandasLikeExpr | bool | Any) -> Self:
        return reuse_series_implementation(self, "__or__", other=other)

    def __ror__(self, other: Any) -> Self:
        other = self.__narwhals_namespace__().lit(other, dtype=None)
        return other.__or__(self)  # type: ignore[no-any-return]

    def __add__(self, other: PandasLikeExpr | Any) -> Self:
        return reuse_series_implementation(self, "__add__", other=other)

    def __radd__(self, other: Any) -> Self:
        other = self.__narwhals_namespace__().lit(other, dtype=None)
        return other.__add__(self)  # type: ignore[no-any-return]

    def __sub__(self, other: PandasLikeExpr | Any) -> Self:
        return reuse_series_implementation(self, "__sub__", other=other)

    def __rsub__(self, other: Any) -> Self:
        other = self.__narwhals_namespace__().lit(other, dtype=None)
        return other.__sub__(self)  # type: ignore[no-any-return]

    def __mul__(self, other: PandasLikeExpr | Any) -> Self:
        return reuse_series_implementation(self, "__mul__", other=other)

    def __rmul__(self, other: Any) -> Self:
        other = self.__narwhals_namespace__().lit(other, dtype=None)
        return other.__mul__(self)  # type: ignore[no-any-return]

    def __truediv__(self, other: PandasLikeExpr | Any) -> Self:
        return reuse_series_implementation(self, "__truediv__", other=other)

    def __rtruediv__(self, other: Any) -> Self:
        other = self.__narwhals_namespace__().lit(other, dtype=None)
        return other.__truediv__(self)  # type: ignore[no-any-return]

    def __floordiv__(self, other: PandasLikeExpr | Any) -> Self:
        return reuse_series_implementation(self, "__floordiv__", other=other)

    def __rfloordiv__(self, other: Any) -> Self:
        other = self.__narwhals_namespace__().lit(other, dtype=None)
        return other.__floordiv__(self)  # type: ignore[no-any-return]

    def __pow__(self, other: PandasLikeExpr | Any) -> Self:
        return reuse_series_implementation(self, "__pow__", other=other)

    def __rpow__(self, other: Any) -> Self:
        other = self.__narwhals_namespace__().lit(other, dtype=None)
        return other.__pow__(self)  # type: ignore[no-any-return]

    def __mod__(self, other: PandasLikeExpr | Any) -> Self:
        return reuse_series_implementation(self, "__mod__", other=other)

    def __rmod__(self, other: Any) -> Self:
        other = self.__narwhals_namespace__().lit(other, dtype=None)
        return other.__mod__(self)  # type: ignore[no-any-return]

    # Unary

    def __invert__(self) -> Self:
        return reuse_series_implementation(self, "__invert__")

    # Reductions
    def null_count(self) -> Self:
        return reuse_series_implementation(self, "null_count", returns_scalar=True)

    def n_unique(self) -> Self:
        return reuse_series_implementation(self, "n_unique", returns_scalar=True)

    def sum(self) -> Self:
        return reuse_series_implementation(self, "sum", returns_scalar=True)

    def count(self) -> Self:
        return reuse_series_implementation(self, "count", returns_scalar=True)

    def mean(self) -> Self:
        return reuse_series_implementation(self, "mean", returns_scalar=True)

    def median(self) -> Self:
        return reuse_series_implementation(self, "median", returns_scalar=True)

    def std(self, *, ddof: int = 1) -> Self:
        return reuse_series_implementation(self, "std", ddof=ddof, returns_scalar=True)

    def skew(self: Self) -> Self:
        return reuse_series_implementation(self, "skew", returns_scalar=True)

    def any(self) -> Self:
        return reuse_series_implementation(self, "any", returns_scalar=True)

    def all(self) -> Self:
        return reuse_series_implementation(self, "all", returns_scalar=True)

    def max(self) -> Self:
        return reuse_series_implementation(self, "max", returns_scalar=True)

    def min(self) -> Self:
        return reuse_series_implementation(self, "min", returns_scalar=True)

    # Other

    def clip(self, lower_bound: Any, upper_bound: Any) -> Self:
        return reuse_series_implementation(
            self, "clip", lower_bound=lower_bound, upper_bound=upper_bound
        )

    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Self:
        return reuse_series_implementation(
            self,
            "is_between",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            closed=closed,
        )

    def is_null(self) -> Self:
        return reuse_series_implementation(self, "is_null")

    def fill_null(
        self,
        value: Any | None = None,
        strategy: Literal["forward", "backward"] | None = None,
        limit: int | None = None,
    ) -> Self:
        return reuse_series_implementation(
            self, "fill_null", value=value, strategy=strategy, limit=limit
        )

    def is_in(self, other: Any) -> Self:
        return reuse_series_implementation(self, "is_in", other=other)

    def arg_true(self) -> Self:
        return reuse_series_implementation(self, "arg_true")

    def ewm_mean(
        self,
        *,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        min_periods: int = 1,
        ignore_nulls: bool = False,
    ) -> Self:
        return reuse_series_implementation(
            self,
            "ewm_mean",
            com=com,
            span=span,
            half_life=half_life,
            alpha=alpha,
            adjust=adjust,
            min_periods=min_periods,
            ignore_nulls=ignore_nulls,
        )

    def filter(self, *predicates: Any) -> Self:
        plx = self.__narwhals_namespace__()
        other = plx.all_horizontal(*predicates)
        return reuse_series_implementation(self, "filter", other=other)

    def drop_nulls(self) -> Self:
        return reuse_series_implementation(self, "drop_nulls")

    def replace_strict(
        self, old: Sequence[Any], new: Sequence[Any], *, return_dtype: DType | None
    ) -> Self:
        return reuse_series_implementation(
            self, "replace_strict", old, new, return_dtype=return_dtype
        )

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        return reuse_series_implementation(
            self, "sort", descending=descending, nulls_last=nulls_last
        )

    def abs(self) -> Self:
        return reuse_series_implementation(self, "abs")

    def cum_sum(self: Self, *, reverse: bool) -> Self:
        return reuse_series_implementation(self, "cum_sum", reverse=reverse)

    def unique(self, *, maintain_order: bool = False) -> Self:
        return reuse_series_implementation(self, "unique", maintain_order=maintain_order)

    def diff(self) -> Self:
        return reuse_series_implementation(self, "diff")

    def shift(self, n: int) -> Self:
        return reuse_series_implementation(self, "shift", n=n)

    def sample(
        self: Self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        with_replacement: bool = False,
        seed: int | None = None,
    ) -> Self:
        return reuse_series_implementation(
            self,
            "sample",
            n=n,
            fraction=fraction,
            with_replacement=with_replacement,
            seed=seed,
        )

    def alias(self, name: str) -> Self:
        # Define this one manually, so that we can
        # override `output_names` and not increase depth
        return self.__class__(
            lambda df: [series.alias(name) for series in self._call(df)],
            depth=self._depth,
            function_name=self._function_name,
            root_names=self._root_names,
            output_names=[name],
            implementation=self._implementation,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def over(self, keys: list[str]) -> Self:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            if self._output_names is None:
                msg = (
                    "Anonymous expressions are not supported in over.\n"
                    "Instead of `nw.all()`, try using a named expression, such as "
                    "`nw.col('a', 'b')`\n"
                )
                raise ValueError(msg)
            tmp = df.group_by(*keys, drop_null_keys=False).agg(self)
            tmp = df.select(*keys).join(
                tmp, how="left", left_on=keys, right_on=keys, suffix="_right"
            )
            return [tmp[name] for name in self._output_names]

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=self._function_name + "->over",
            root_names=self._root_names,
            output_names=self._output_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def is_duplicated(self) -> Self:
        return reuse_series_implementation(self, "is_duplicated")

    def is_unique(self) -> Self:
        return reuse_series_implementation(self, "is_unique")

    def is_first_distinct(self) -> Self:
        return reuse_series_implementation(self, "is_first_distinct")

    def is_last_distinct(self) -> Self:
        return reuse_series_implementation(self, "is_last_distinct")

    def quantile(
        self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Self:
        return reuse_series_implementation(
            self, "quantile", quantile, interpolation, returns_scalar=True
        )

    def head(self, n: int) -> Self:
        return reuse_series_implementation(self, "head", n)

    def tail(self, n: int) -> Self:
        return reuse_series_implementation(self, "tail", n)

    def round(self: Self, decimals: int) -> Self:
        return reuse_series_implementation(self, "round", decimals)

    def len(self: Self) -> Self:
        return reuse_series_implementation(self, "len", returns_scalar=True)

    def gather_every(self: Self, n: int, offset: int = 0) -> Self:
        return reuse_series_implementation(self, "gather_every", n=n, offset=offset)

    def mode(self: Self) -> Self:
        return reuse_series_implementation(self, "mode")

    def map_batches(
        self: Self,
        function: Callable[[Any], Any],
        return_dtype: DType | None = None,
    ) -> Self:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            input_series_list = self._call(df)
            output_names = [input_series.name for input_series in input_series_list]
            result = [function(series) for series in input_series_list]
            if is_numpy_array(result[0]) or (
                (np := get_numpy()) is not None and np.isscalar(result[0])
            ):
                result = [
                    df.__narwhals_namespace__()
                    ._create_compliant_series(array)
                    .alias(output_name)
                    for array, output_name in zip(result, output_names)
                ]
            if return_dtype is not None:
                result = [series.cast(return_dtype) for series in result]
            return result

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=self._function_name + "->map_batches",
            root_names=self._root_names,
            output_names=self._output_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def is_finite(self: Self) -> Self:
        return reuse_series_implementation(self, "is_finite")

    def cum_count(self: Self, *, reverse: bool) -> Self:
        return reuse_series_implementation(self, "cum_count", reverse=reverse)

    def cum_min(self: Self, *, reverse: bool) -> Self:
        return reuse_series_implementation(self, "cum_min", reverse=reverse)

    def cum_max(self: Self, *, reverse: bool) -> Self:
        return reuse_series_implementation(self, "cum_max", reverse=reverse)

    def cum_prod(self: Self, *, reverse: bool) -> Self:
        return reuse_series_implementation(self, "cum_prod", reverse=reverse)

    def rolling_sum(
        self: Self,
        window_size: int,
        *,
        min_periods: int | None,
        center: bool,
    ) -> Self:
        return reuse_series_implementation(
            self,
            "rolling_sum",
            window_size=window_size,
            min_periods=min_periods,
            center=center,
        )

    def rolling_mean(
        self: Self,
        window_size: int,
        *,
        min_periods: int | None,
        center: bool,
    ) -> Self:
        return reuse_series_implementation(
            self,
            "rolling_mean",
            window_size=window_size,
            min_periods=min_periods,
            center=center,
        )

    @property
    def str(self: Self) -> PandasLikeExprStringNamespace:
        return PandasLikeExprStringNamespace(self)

    @property
    def dt(self: Self) -> PandasLikeExprDateTimeNamespace:
        return PandasLikeExprDateTimeNamespace(self)

    @property
    def cat(self: Self) -> PandasLikeExprCatNamespace:
        return PandasLikeExprCatNamespace(self)

    @property
    def name(self: Self) -> PandasLikeExprNameNamespace:
        return PandasLikeExprNameNamespace(self)


class PandasLikeExprCatNamespace:
    def __init__(self, expr: PandasLikeExpr) -> None:
        self._expr = expr

    def get_categories(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "cat",
            "get_categories",
        )


class PandasLikeExprStringNamespace:
    def __init__(self, expr: PandasLikeExpr) -> None:
        self._expr = expr

    def len_chars(
        self,
    ) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "str", "len_chars")

    def replace(
        self,
        pattern: str,
        value: str,
        *,
        literal: bool = False,
        n: int = 1,
    ) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr, "str", "replace", pattern, value, literal=literal, n=n
        )

    def replace_all(
        self,
        pattern: str,
        value: str,
        *,
        literal: bool = False,
    ) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr, "str", "replace_all", pattern, value, literal=literal
        )

    def strip_chars(self, characters: str | None = None) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "strip_chars",
            characters,
        )

    def starts_with(self, prefix: str) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "starts_with",
            prefix,
        )

    def ends_with(self, suffix: str) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "ends_with",
            suffix,
        )

    def contains(self, pattern: str, *, literal: bool) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "contains",
            pattern,
            literal=literal,
        )

    def slice(self, offset: int, length: int | None = None) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr, "str", "slice", offset, length
        )

    def to_datetime(self: Self, format: str | None) -> PandasLikeExpr:  # noqa: A002
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "to_datetime",
            format,
        )

    def to_uppercase(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "to_uppercase",
        )

    def to_lowercase(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "to_lowercase",
        )


class PandasLikeExprDateTimeNamespace:
    def __init__(self, expr: PandasLikeExpr) -> None:
        self._expr = expr

    def date(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "date")

    def year(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "year")

    def month(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "month")

    def day(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "day")

    def hour(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "hour")

    def minute(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "minute")

    def second(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "second")

    def millisecond(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "millisecond")

    def microsecond(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "microsecond")

    def nanosecond(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "nanosecond")

    def ordinal_day(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "ordinal_day")

    def total_minutes(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "total_minutes")

    def total_seconds(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "total_seconds")

    def total_milliseconds(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr, "dt", "total_milliseconds"
        )

    def total_microseconds(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr, "dt", "total_microseconds"
        )

    def total_nanoseconds(self) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr, "dt", "total_nanoseconds"
        )

    def to_string(self, format: str) -> PandasLikeExpr:  # noqa: A002
        return reuse_series_namespace_implementation(
            self._expr, "dt", "to_string", format
        )

    def replace_time_zone(self, time_zone: str | None) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr, "dt", "replace_time_zone", time_zone
        )

    def convert_time_zone(self, time_zone: str) -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr, "dt", "convert_time_zone", time_zone
        )

    def timestamp(self, time_unit: Literal["ns", "us", "ms"] = "us") -> PandasLikeExpr:
        return reuse_series_namespace_implementation(
            self._expr, "dt", "timestamp", time_unit
        )


class PandasLikeExprNameNamespace:
    def __init__(self: Self, expr: PandasLikeExpr) -> None:
        self._expr = expr

    def keep(self: Self) -> PandasLikeExpr:
        root_names = self._expr._root_names

        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.keep`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)

        return self._expr.__class__(
            lambda df: [
                series.alias(name)
                for series, name in zip(self._expr._call(df), root_names)
            ],
            depth=self._expr._depth,
            function_name=self._expr._function_name,
            root_names=root_names,
            output_names=root_names,
            implementation=self._expr._implementation,
            backend_version=self._expr._backend_version,
            dtypes=self._expr._dtypes,
        )

    def map(self: Self, function: Callable[[str], str]) -> PandasLikeExpr:
        root_names = self._expr._root_names

        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.map`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)

        output_names = [function(str(name)) for name in root_names]

        return self._expr.__class__(
            lambda df: [
                series.alias(name)
                for series, name in zip(self._expr._call(df), output_names)
            ],
            depth=self._expr._depth,
            function_name=self._expr._function_name,
            root_names=root_names,
            output_names=output_names,
            implementation=self._expr._implementation,
            backend_version=self._expr._backend_version,
            dtypes=self._expr._dtypes,
        )

    def prefix(self: Self, prefix: str) -> PandasLikeExpr:
        root_names = self._expr._root_names
        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.prefix`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)

        output_names = [prefix + str(name) for name in root_names]
        return self._expr.__class__(
            lambda df: [
                series.alias(name)
                for series, name in zip(self._expr._call(df), output_names)
            ],
            depth=self._expr._depth,
            function_name=self._expr._function_name,
            root_names=root_names,
            output_names=output_names,
            implementation=self._expr._implementation,
            backend_version=self._expr._backend_version,
            dtypes=self._expr._dtypes,
        )

    def suffix(self: Self, suffix: str) -> PandasLikeExpr:
        root_names = self._expr._root_names
        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.suffix`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)

        output_names = [str(name) + suffix for name in root_names]

        return self._expr.__class__(
            lambda df: [
                series.alias(name)
                for series, name in zip(self._expr._call(df), output_names)
            ],
            depth=self._expr._depth,
            function_name=self._expr._function_name,
            root_names=root_names,
            output_names=output_names,
            implementation=self._expr._implementation,
            backend_version=self._expr._backend_version,
            dtypes=self._expr._dtypes,
        )

    def to_lowercase(self: Self) -> PandasLikeExpr:
        root_names = self._expr._root_names

        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.to_lowercase`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)
        output_names = [str(name).lower() for name in root_names]

        return self._expr.__class__(
            lambda df: [
                series.alias(name)
                for series, name in zip(self._expr._call(df), output_names)
            ],
            depth=self._expr._depth,
            function_name=self._expr._function_name,
            root_names=root_names,
            output_names=output_names,
            implementation=self._expr._implementation,
            backend_version=self._expr._backend_version,
            dtypes=self._expr._dtypes,
        )

    def to_uppercase(self: Self) -> PandasLikeExpr:
        root_names = self._expr._root_names

        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.to_uppercase`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)
        output_names = [str(name).upper() for name in root_names]

        return self._expr.__class__(
            lambda df: [
                series.alias(name)
                for series, name in zip(self._expr._call(df), output_names)
            ],
            depth=self._expr._depth,
            function_name=self._expr._function_name,
            root_names=root_names,
            output_names=output_names,
            implementation=self._expr._implementation,
            backend_version=self._expr._backend_version,
            dtypes=self._expr._dtypes,
        )
