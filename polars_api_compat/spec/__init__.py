from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Protocol

from typing_extensions import Self


class Expr(Protocol):
    call: Callable[[DataFrame | LazyFrame], list[Series]]
    api_version: str
    depth: int | None
    root_names: list[str] | None
    output_names: list[str] | None
    function_name: str | None

    def alias(self, name: str) -> Expr:
        ...

    def __expr_namespace__(self) -> Namespace:
        ...

    def __and__(self, other: Any) -> Expr:
        ...

    def __or__(self, other: Any) -> Expr:
        ...

    def __add__(self, other: Any) -> Expr:
        ...

    def __radd__(self, other: Any) -> Expr:
        ...

    def __sub__(self, other: Any) -> Expr:
        ...

    def __rsub__(self, other: Any) -> Expr:
        ...

    def __mul__(self, other: Any) -> Expr:
        ...

    def __rmul__(self, other: Any) -> Expr:
        ...

    def __le__(self, other: Any) -> Expr:
        ...

    def mean(self) -> Expr:
        ...

    def sum(self) -> Expr:
        ...


class Namespace(Protocol):
    def col(self, *names: str | Iterable[str]) -> Expr:
        ...

    def all(self) -> Expr:
        ...

    def sum(self, *columns: str) -> Expr:
        ...

    def mean(self, *columns: str) -> Expr:
        ...

    def len(self) -> Expr:
        ...

    def _create_series_from_scalar(self, value: Any, series: Series) -> Series:
        ...

    def _create_expr_from_series(self, series: Series) -> Expr:
        ...

    def _create_expr_from_callable(  # noqa: PLR0913
        self,
        func: Callable[[DataFrame | LazyFrame], list[Series]],
        depth: int,
        function_name: str | None,
        root_names: list[str] | None,
        output_names: list[str] | None,
    ) -> Expr:
        ...

    def all_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
        ...

    def any_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
        ...

    def sum_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
        ...


class Series(Protocol):
    def __series_namespace__(self) -> Namespace:
        ...

    def alias(self, name: str) -> Self:
        ...

    @property
    def series(self) -> Any:
        """
        Return the underlying Series.

        This is typically what you'll want to return at the end
        of a series-agnostic function.
        """
        ...

    @property
    def name(self) -> str:
        ...


class DataFrame(Protocol):
    api_version: str
    columns: list[str]

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> DataFrame:
        ...

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> DataFrame:
        ...

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> DataFrame:
        ...

    def sort(
        self, *keys: str | Iterable[str], descending: bool | Iterable[bool] = False
    ) -> DataFrame:
        ...

    @property
    def dataframe(self) -> Any:
        """
        Return the underlying DataFrame.

        This is typically what you'll want to return at the end
        of a dataframe-agnostic function.
        """
        ...

    def group_by(self, *keys: str | Iterable[str]) -> GroupBy:
        ...

    def __dataframe_namespace__(self) -> Namespace:
        ...

    def lazy(self) -> LazyFrame:
        ...

    def join(
        self,
        other: DataFrame,
        *,
        how: Literal["left", "inner", "outer"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> DataFrame:
        ...


class LazyFrame(Protocol):
    api_version: str
    columns: list[str]

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> LazyFrame:
        ...

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> LazyFrame:
        ...

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> LazyFrame:
        ...

    def sort(
        self, *keys: str | Iterable[str], descending: bool | Iterable[bool] = False
    ) -> LazyFrame:
        ...

    def collect(self) -> DataFrame:
        ...

    @property
    def dataframe(self) -> Any:
        """
        Return the underlying DataFrame.

        This is typically what you'll want to return at the end
        of a dataframe-agnostic function.
        """
        ...

    def group_by(self, *keys: str | Iterable[str]) -> LazyGroupBy:
        ...

    def __lazyframe_namespace__(self) -> Namespace:
        ...

    def join(
        self,
        other: LazyFrame,
        *,
        how: Literal["left", "inner", "outer"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> LazyFrame:
        ...


class GroupBy(Protocol):
    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> DataFrame:
        ...


class LazyGroupBy(Protocol):
    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> LazyFrame:
        ...


IntoExpr = Expr | str | int | float | Series
