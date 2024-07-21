from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from narwhals._expression_parsing import reuse_series_implementation
from narwhals._expression_parsing import reuse_series_namespace_implementation

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals._arrow.series import ArrowSeries
    from narwhals.dtypes import DType


class ArrowExpr:
    def __init__(
        self,
        call: Callable[[ArrowDataFrame], list[ArrowSeries]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        backend_version: tuple[int, ...],
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._depth = depth
        self._output_names = output_names
        self._implementation = "arrow"
        self._backend_version = backend_version

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ArrowExpr("
            f"depth={self._depth}, "
            f"function_name={self._function_name}, "
            f"root_names={self._root_names}, "
            f"output_names={self._output_names}"
        )

    @classmethod
    def from_column_names(
        cls: type[Self], *column_names: str, backend_version: tuple[int, ...]
    ) -> Self:
        from narwhals._arrow.series import ArrowSeries

        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            return [
                ArrowSeries(
                    df._native_dataframe[column_name],
                    name=column_name,
                    backend_version=df._backend_version,
                )
                for column_name in column_names
            ]

        return cls(
            func,
            depth=0,
            function_name="col",
            root_names=list(column_names),
            output_names=list(column_names),
            backend_version=backend_version,
        )

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._arrow.namespace import ArrowNamespace

        return ArrowNamespace(backend_version=self._backend_version)

    def __narwhals_expr__(self) -> None: ...

    def __eq__(self, other: ArrowExpr | Any) -> Self:  # type: ignore[override]
        return reuse_series_implementation(self, "__eq__", other=other)

    def __ne__(self, other: ArrowExpr | Any) -> Self:  # type: ignore[override]
        return reuse_series_implementation(self, "__ne__", other=other)

    def __ge__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__ge__", other=other)

    def __gt__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__gt__", other=other)

    def __le__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__le__", other=other)

    def __lt__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__lt__", other=other)

    def __and__(self, other: ArrowExpr | bool | Any) -> Self:
        return reuse_series_implementation(self, "__and__", other=other)

    def __or__(self, other: ArrowExpr | bool | Any) -> Self:
        return reuse_series_implementation(self, "__or__", other=other)

    def __add__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__add__", other)

    def __radd__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__radd__", other)

    def __sub__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__sub__", other)

    def __rsub__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__rsub__", other)

    def __mul__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__mul__", other)

    def __rmul__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__rmul__", other)

    def __pow__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__pow__", other)

    def __rpow__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__rpow__", other)

    def __floordiv__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__floordiv__", other)

    def __rfloordiv__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__rfloordiv__", other)

    def __truediv__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__truediv__", other)

    def __rtruediv__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__rtruediv__", other)

    def __invert__(self) -> Self:
        return reuse_series_implementation(self, "__invert__")

    def len(self) -> Self:
        return reuse_series_implementation(self, "len", returns_scalar=True)

    def filter(self, *predicates: Any) -> Self:
        from narwhals._arrow.namespace import ArrowNamespace

        plx = ArrowNamespace(backend_version=self._backend_version)
        expr = plx.all_horizontal(*predicates)
        return reuse_series_implementation(self, "filter", other=expr)

    def mean(self) -> Self:
        return reuse_series_implementation(self, "mean", returns_scalar=True)

    def count(self) -> Self:
        return reuse_series_implementation(self, "count", returns_scalar=True)

    def n_unique(self) -> Self:
        return reuse_series_implementation(self, "n_unique", returns_scalar=True)

    def std(self, ddof: int = 1) -> Self:
        return reuse_series_implementation(self, "std", ddof=ddof, returns_scalar=True)

    def cast(self, dtype: DType) -> Self:
        return reuse_series_implementation(self, "cast", dtype)

    def abs(self) -> Self:
        return reuse_series_implementation(self, "abs")

    def diff(self) -> Self:
        return reuse_series_implementation(self, "diff")

    def cum_sum(self) -> Self:
        return reuse_series_implementation(self, "cum_sum")

    def any(self) -> Self:
        return reuse_series_implementation(self, "any", returns_scalar=True)

    def min(self) -> Self:
        return reuse_series_implementation(self, "min", returns_scalar=True)

    def max(self) -> Self:
        return reuse_series_implementation(self, "max", returns_scalar=True)

    def all(self) -> Self:
        return reuse_series_implementation(self, "all", returns_scalar=True)

    def sum(self) -> Self:
        return reuse_series_implementation(self, "sum", returns_scalar=True)

    def alias(self, name: str) -> Self:
        # Define this one manually, so that we can
        # override `output_names` and not increase depth
        return self.__class__(
            lambda df: [series.alias(name) for series in self._call(df)],
            depth=self._depth,
            function_name=self._function_name,
            root_names=self._root_names,
            output_names=[name],
            backend_version=self._backend_version,
        )

    def null_count(self) -> Self:
        return reuse_series_implementation(self, "null_count", returns_scalar=True)

    def is_null(self) -> Self:
        return reuse_series_implementation(self, "is_null")

    def head(self, n: int) -> Self:
        return reuse_series_implementation(self, "head", n)

    def tail(self, n: int) -> Self:
        return reuse_series_implementation(self, "tail", n)

    def is_in(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "is_in", other)

    def sample(
        self: Self,
        n: int | None = None,
        fraction: float | None = None,
        *,
        with_replacement: bool = False,
    ) -> Self:
        return reuse_series_implementation(
            self, "sample", n=n, fraction=fraction, with_replacement=with_replacement
        )

    def fill_null(self: Self, value: Any) -> Self:
        return reuse_series_implementation(self, "fill_null", value=value)

    @property
    def dt(self: Self) -> ArrowExprDateTimeNamespace:
        return ArrowExprDateTimeNamespace(self)

    @property
    def str(self: Self) -> ArrowExprStringNamespace:
        return ArrowExprStringNamespace(self)

    @property
    def cat(self: Self) -> ArrowExprCatNamespace:
        return ArrowExprCatNamespace(self)

    @property
    def name(self: Self) -> ArrowExprNameNamespace:
        return ArrowExprNameNamespace(self)


class ArrowExprCatNamespace:
    def __init__(self, expr: ArrowExpr) -> None:
        self._expr = expr

    def get_categories(self) -> ArrowExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "cat",
            "get_categories",
        )


class ArrowExprDateTimeNamespace:
    def __init__(self, expr: ArrowExpr) -> None:
        self._expr = expr

    def to_string(self, format: str) -> ArrowExpr:  # noqa: A002
        return reuse_series_namespace_implementation(
            self._expr, "dt", "to_string", format
        )


class ArrowExprStringNamespace:
    def __init__(self, expr: ArrowExpr) -> None:
        self._expr = expr

    def starts_with(self, prefix: str) -> ArrowExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "starts_with",
            prefix,
        )

    def ends_with(self, suffix: str) -> ArrowExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "ends_with",
            suffix,
        )

    def contains(self, pattern: str, *, literal: bool) -> ArrowExpr:
        return reuse_series_namespace_implementation(
            self._expr, "str", "contains", pattern, literal=literal
        )

    def slice(self, offset: int, length: int | None = None) -> ArrowExpr:
        return reuse_series_namespace_implementation(
            self._expr, "str", "slice", offset, length
        )

    def to_datetime(self, format: str | None = None) -> ArrowExpr:  # noqa: A002
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "to_datetime",
            format,
        )

    def to_uppercase(self) -> ArrowExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "to_uppercase",
        )

    def to_lowercase(self) -> ArrowExpr:
        return reuse_series_namespace_implementation(
            self._expr,
            "str",
            "to_lowercase",
        )


class ArrowExprNameNamespace:
    def __init__(self: Self, expr: ArrowExpr) -> None:
        self._expr = expr

    def keep(self: Self) -> ArrowExpr:
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
            backend_version=self._expr._backend_version,
        )

    def map(self: Self, function: Callable[[str], str]) -> ArrowExpr:
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
            backend_version=self._expr._backend_version,
        )

    def prefix(self: Self, prefix: str) -> ArrowExpr:
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
            backend_version=self._expr._backend_version,
        )

    def suffix(self: Self, suffix: str) -> ArrowExpr:
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
            backend_version=self._expr._backend_version,
        )

    def to_lowercase(self: Self) -> ArrowExpr:
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
            backend_version=self._expr._backend_version,
        )

    def to_uppercase(self: Self) -> ArrowExpr:
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
            backend_version=self._expr._backend_version,
        )
