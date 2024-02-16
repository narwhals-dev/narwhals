from __future__ import annotations
from typing import Protocol, Iterable, Any

from typing_extensions import Self

class Expr(Protocol):
    def alias(self, name: str) -> Self:
        ...

    def call(self, df: DataFrame | LazyFrame) -> list[Series]:
        ...


class Namespace(Protocol):
    def col(self, *names: str | Iterable[str]) -> Expr:
        ...

class Series(Protocol):
    def alias(self, name: str) -> Self:
        ...

class DataFrame(Protocol):
    def with_columns(self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> Self:
        ...

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        ...

    def select(self, *exprs: Expr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> Self:
        ...

    @property
    def dataframe(self) -> Any:
        """
        Return the underlying DataFrame.

        This is typically what you'll want to return at the end
        of a dataframe-agnostic function.
        """
        ...
    
    def __dataframe_namespace__(self) -> Namespace:
        ...

class LazyFrame(Protocol):
    def with_columns(self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> Self:
        ...

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        ...

    def select(self, *exprs: Expr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> Self:
        ...

    def collect(self) -> DataFrame:
        ...

    def __lazyframe_namespace__(self) -> Namespace:
        ...

IntoExpr = Expr | str | int | float | Series