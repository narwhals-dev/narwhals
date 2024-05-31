from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal

from narwhals._pandas_like.series import PandasSeries
from narwhals._pandas_like.utils import reuse_series_implementation
from narwhals._pandas_like.utils import reuse_series_namespace_implementation

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.dataframe import PandasDataFrame


class PandasExpr:
    def __init__(  # noqa: PLR0913
        self,
        call: Callable[[PandasDataFrame], list[PandasSeries]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        implementation: str,
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._depth = depth
        self._output_names = output_names
        self._implementation = implementation

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PandasExpr("
            f"depth={self._depth}, "
            f"function_name={self._function_name}, "
            f"root_names={self._root_names}, "
            f"output_names={self._output_names}"
        )

    @classmethod
    def from_column_names(
        cls: type[Self], *column_names: str, implementation: str
    ) -> Self:
        def func(df: PandasDataFrame) -> list[PandasSeries]:
            return [
                PandasSeries(
                    df._dataframe.loc[:, column_name],
                    implementation=df._implementation,
                )
                for column_name in column_names
            ]

        return cls(
            func,
            depth=0,
            function_name="col",
            root_names=list(column_names),
            output_names=list(column_names),
            implementation=implementation,
        )

    def cast(
        self,
        dtype: Any,
    ) -> Self:
        return reuse_series_implementation(self, "cast", dtype=dtype)

    def __eq__(self, other: PandasExpr | Any) -> Self:  # type: ignore[override]
        return reuse_series_implementation(self, "__eq__", other=other)

    def __ne__(self, other: PandasExpr | Any) -> Self:  # type: ignore[override]
        return reuse_series_implementation(self, "__ne__", other=other)

    def __ge__(self, other: PandasExpr | Any) -> Self:
        return reuse_series_implementation(self, "__ge__", other=other)

    def __gt__(self, other: PandasExpr | Any) -> Self:
        return reuse_series_implementation(self, "__gt__", other=other)

    def __le__(self, other: PandasExpr | Any) -> Self:
        return reuse_series_implementation(self, "__le__", other=other)

    def __lt__(self, other: PandasExpr | Any) -> Self:
        return reuse_series_implementation(self, "__lt__", other=other)

    def __and__(self, other: PandasExpr | bool | Any) -> Self:
        return reuse_series_implementation(self, "__and__", other=other)

    def __rand__(self, other: Any) -> Self:
        return reuse_series_implementation(self, "__rand__", other=other)

    def __or__(self, other: PandasExpr | bool | Any) -> Self:
        return reuse_series_implementation(self, "__or__", other=other)

    def __ror__(self, other: Any) -> Self:
        return reuse_series_implementation(self, "__ror__", other=other)

    def __add__(self, other: PandasExpr | Any) -> Self:
        return reuse_series_implementation(self, "__add__", other=other)

    def __radd__(self, other: Any) -> Self:
        return reuse_series_implementation(self, "__radd__", other=other)

    def __sub__(self, other: PandasExpr | Any) -> Self:
        return reuse_series_implementation(self, "__sub__", other=other)

    def __rsub__(self, other: Any) -> Self:
        return reuse_series_implementation(self, "__rsub__", other=other)

    def __mul__(self, other: PandasExpr | Any) -> Self:
        return reuse_series_implementation(self, "__mul__", other=other)

    def __rmul__(self, other: Any) -> Self:
        return reuse_series_implementation(self, "__rmul__", other=other)

    def __truediv__(self, other: PandasExpr | Any) -> Self:
        return reuse_series_implementation(self, "__truediv__", other=other)

    def __rtruediv__(self, other: Any) -> Self:
        return reuse_series_implementation(self, "__rtruediv__", other=other)

    def __floordiv__(self, other: PandasExpr | Any) -> Self:
        return reuse_series_implementation(self, "__floordiv__", other=other)

    def __rfloordiv__(self, other: Any) -> Self:
        return reuse_series_implementation(self, "__rfloordiv__", other=other)

    def __pow__(self, other: PandasExpr | Any) -> Self:
        return reuse_series_implementation(self, "__pow__", other=other)

    def __rpow__(self, other: Any) -> Self:
        return reuse_series_implementation(self, "__rpow__", other=other)

    def __mod__(self, other: PandasExpr | Any) -> Self:
        return reuse_series_implementation(self, "__mod__", other=other)

    def __rmod__(self, other: Any) -> Self:
        return reuse_series_implementation(self, "__rmod__", other=other)

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

    def mean(self) -> Self:
        return reuse_series_implementation(self, "mean", returns_scalar=True)

    def std(self, *, ddof: int = 1) -> Self:
        return reuse_series_implementation(self, "std", ddof=ddof, returns_scalar=True)

    def any(self) -> Self:
        return reuse_series_implementation(self, "any", returns_scalar=True)

    def all(self) -> Self:
        return reuse_series_implementation(self, "all", returns_scalar=True)

    def max(self) -> Self:
        return reuse_series_implementation(self, "max", returns_scalar=True)

    def min(self) -> Self:
        return reuse_series_implementation(self, "min", returns_scalar=True)

    # Other
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

    def fill_null(self, value: Any) -> Self:
        return reuse_series_implementation(self, "fill_null", value=value)

    def is_in(self, other: Any) -> Self:
        return reuse_series_implementation(self, "is_in", other=other)

    def filter(self, *predicates: Any) -> Self:
        from narwhals._pandas_like.namespace import PandasNamespace

        plx = PandasNamespace(self._implementation)
        expr = plx.all_horizontal(*predicates)
        return reuse_series_implementation(self, "filter", other=expr)

    def drop_nulls(self) -> Self:
        return reuse_series_implementation(self, "drop_nulls")

    def sort(self, *, descending: bool = False) -> Self:
        return reuse_series_implementation(self, "sort", descending=descending)

    def cum_sum(self) -> Self:
        return reuse_series_implementation(self, "cum_sum")

    def unique(self) -> Self:
        return reuse_series_implementation(self, "unique")

    def diff(self) -> Self:
        return reuse_series_implementation(self, "diff")

    def shift(self, n: int) -> Self:
        return reuse_series_implementation(self, "shift", n=n)

    def sample(
        self,
        n: int | None = None,
        fraction: float | None = None,
        *,
        with_replacement: bool = False,
    ) -> Self:
        return reuse_series_implementation(
            self, "sample", n=n, fraction=fraction, with_replacement=with_replacement
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
        )

    def over(self, keys: list[str]) -> Self:
        def func(df: PandasDataFrame) -> list[PandasSeries]:
            if self._output_names is None:
                msg = (
                    "Anonymous expressions are not supported in over.\n"
                    "Instead of `nw.all()`, try using a named expression, such as "
                    "`nw.col('a', 'b')`\n"
                )
                raise ValueError(msg)
            tmp = df.group_by(keys).agg(self)
            tmp = df.select(keys).join(tmp, how="left", left_on=keys, right_on=keys)
            return [tmp[name] for name in self._output_names]

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=self._function_name + "->over",
            root_names=self._root_names,
            output_names=self._output_names,
            implementation=self._implementation,
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

    @property
    def str(self) -> PandasExprStringNamespace:
        return PandasExprStringNamespace(self)

    @property
    def dt(self) -> PandasExprDateTimeNamespace:
        return PandasExprDateTimeNamespace(self)


class PandasExprStringNamespace:
    def __init__(self, expr: PandasExpr) -> None:
        self._expr = expr

    def ends_with(self, suffix: str) -> PandasExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "ends_with",
            suffix,
        )

    def head(self, n: int = 5) -> PandasExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "head",
            n,
        )

    def to_datetime(self, format: str | None = None) -> PandasExpr:  # noqa: A002
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "to_datetime",
            format,
        )


class PandasExprDateTimeNamespace:
    def __init__(self, expr: PandasExpr) -> None:
        self._expr = expr

    def year(self) -> PandasExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "year")

    def month(self) -> PandasExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "month")

    def day(self) -> PandasExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "day")

    def hour(self) -> PandasExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "hour")

    def minute(self) -> PandasExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "minute")

    def second(self) -> PandasExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "second")

    def millisecond(self) -> PandasExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "millisecond")

    def microsecond(self) -> PandasExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "microsecond")

    def nanosecond(self) -> PandasExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "nanosecond")

    def ordinal_day(self) -> PandasExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "ordinal_day")

    def total_minutes(self) -> PandasExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "total_minutes")

    def total_seconds(self) -> PandasExpr:
        return reuse_series_namespace_implementation(self._expr, "dt", "total_seconds")

    def total_milliseconds(self) -> PandasExpr:
        return reuse_series_namespace_implementation(
            self._expr, "dt", "total_milliseconds"
        )

    def total_microseconds(self) -> PandasExpr:
        return reuse_series_namespace_implementation(
            self._expr, "dt", "total_microseconds"
        )

    def total_nanoseconds(self) -> PandasExpr:
        return reuse_series_namespace_implementation(
            self._expr, "dt", "total_nanoseconds"
        )
