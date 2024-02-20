from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Protocol
from typing import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self


class Expr(Protocol):
    def alias(self, name: str) -> Expr:
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

    def __lt__(self, other: Any) -> Expr:
        ...

    def __gt__(self, other: Any) -> Expr:
        ...

    def __ge__(self, other: Any) -> Expr:
        ...

    def mean(self) -> Expr:
        ...

    def sum(self) -> Expr:
        ...

    def min(self) -> Expr:
        ...

    def max(self) -> Expr:
        ...

    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Expr:
        ...

    def is_in(self, other: Any) -> Expr:
        ...

    def is_null(self) -> Expr:
        ...

    def drop_nulls(self) -> Expr:
        ...

    def n_unique(self) -> Expr:
        ...

    def sample(self, n: int, fraction: float, *, with_replacement: bool) -> Expr:
        ...


class ExprStringNamespace(Protocol):
    def ends_with(self, other: str) -> Expr:
        ...


class ExprNameNamespace(Protocol):
    def keep(self) -> Expr:
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

    def all_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
        ...

    def any_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
        ...

    def sum_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
        ...

    def concat(self, items: Iterable[AnyDataFrame], *, how: str) -> AnyDataFrame:
        ...


class Series(Protocol):
    def alias(self, name: str) -> Self:
        ...

    @property
    def name(self) -> str:
        ...

    def item(self) -> Any:
        ...

    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Series:
        ...

    def is_in(self, other: Any) -> Series:
        ...

    def is_null(self) -> Series:
        ...

    def drop_nulls(self) -> Series:
        ...

    def n_unique(self) -> int:
        ...

    def zip_with(self, mask: Series, other: Series) -> Series:
        ...

    def sample(self, n: int, fraction: float, *, with_replacement: bool) -> Series:
        ...

    def to_numpy(self) -> Any:
        ...

    def to_pandas(self) -> Any:
        ...


class DataFrame(Protocol):
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
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Iterable[bool] = False,
    ) -> DataFrame:
        ...

    def group_by(self, *keys: str | Iterable[str]) -> GroupBy:
        ...

    def lazy(self) -> LazyFrame:
        ...

    def join(
        self,
        other: DataFrame,
        *,
        how: Literal["inner"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> DataFrame:
        ...

    @property
    def columns(self) -> list[str]:
        ...

    def head(self, n: int) -> DataFrame:
        ...

    def unique(self, subset: list[str]) -> DataFrame:
        ...

    @property
    def shape(self) -> tuple[int, int]:
        ...

    def rename(self, mapping: dict[str, str]) -> DataFrame:
        ...

    def to_numpy(self) -> Any:
        ...

    def to_pandas(self) -> Any:
        ...


class LazyFrame(Protocol):
    @property
    def columns(self) -> list[str]:
        ...

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
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Iterable[bool] = False,
    ) -> LazyFrame:
        ...

    def collect(self) -> DataFrame:
        ...

    def group_by(self, *keys: str | Iterable[str]) -> LazyGroupBy:
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

    def cache(self) -> LazyFrame:
        ...

    def head(self, n: int) -> LazyFrame:
        ...

    def unique(self, subset: list[str]) -> LazyFrame:
        ...

    def rename(self, mapping: dict[str, str]) -> LazyFrame:
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

AnyDataFrame = TypeVar("AnyDataFrame", DataFrame, LazyFrame)
