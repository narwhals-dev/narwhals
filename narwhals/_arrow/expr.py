from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from narwhals._pandas_like.utils import reuse_series_implementation
from narwhals._pandas_like.utils import reuse_series_namespace_implementation

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals._arrow.series import ArrowSeries
    from narwhals.dtypes import DType


class ArrowExpr:
    def __init__(  # noqa: PLR0913
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

    def __add__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__add__", other)  # type: ignore[type-var]

    def __sub__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__sub__", other)  # type: ignore[type-var]

    def __mul__(self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__mul__", other)  # type: ignore[type-var]

    def mean(self) -> Self:
        return reuse_series_implementation(self, "mean", returns_scalar=True)  # type: ignore[type-var]

    def std(self, ddof: int = 1) -> Self:
        return reuse_series_implementation(self, "std", ddof=ddof, returns_scalar=True)  # type: ignore[type-var]

    def cast(self, dtype: DType) -> Self:
        return reuse_series_implementation(self, "cast", dtype)  # type: ignore[type-var]

    def abs(self) -> Self:
        return reuse_series_implementation(self, "abs")  # type: ignore[type-var]

    def diff(self) -> Self:
        return reuse_series_implementation(self, "diff")  # type: ignore[type-var]

    def cum_sum(self) -> Self:
        return reuse_series_implementation(self, "cum_sum")  # type: ignore[type-var]

    def any(self) -> Self:
        return reuse_series_implementation(self, "any", returns_scalar=True)  # type: ignore[type-var]

    def all(self) -> Self:
        return reuse_series_implementation(self, "all", returns_scalar=True)  # type: ignore[type-var]

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

    @property
    def dt(self) -> ArrowExprDateTimeNamespace:
        return ArrowExprDateTimeNamespace(self)


class ArrowExprDateTimeNamespace:
    def __init__(self, expr: ArrowExpr) -> None:
        self._expr = expr

    def to_string(self, format: str) -> ArrowExpr:  # noqa: A002
        return reuse_series_namespace_implementation(  # type: ignore[type-var, return-value]
            self._expr, "dt", "to_string", format
        )
