from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Protocol
from typing import Sequence
from typing import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self


class Expr(Protocol):
    # --- convert ---
    def alias(self, name: str) -> Self:
        ...

    def cast(self, dtype: DType) -> Self:
        ...

    # --- binary ---
    def __eq__(self, other: object) -> Expr:  # type: ignore[override]
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

    # --- unary ---
    def mean(self) -> Expr:
        ...

    def sum(self) -> Expr:
        ...

    def min(self) -> Expr:
        ...

    def max(self) -> Expr:
        ...

    def n_unique(self) -> Expr:
        ...

    # --- transform ---
    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Expr:
        ...

    def is_in(self, other: Any) -> Expr:
        ...

    def is_null(self) -> Expr:
        ...

    # --- partial reduction ---
    def drop_nulls(self) -> Expr:
        ...

    def sample(self, n: int, fraction: float, *, with_replacement: bool) -> Expr:
        ...

    # --- namespaces ---
    @property
    def str(self) -> ExprStringNamespace:
        ...


class ExprStringNamespace(Protocol):
    def ends_with(self, suffix: str) -> Expr:
        ...


class DType(Protocol):
    @classmethod
    def is_numeric(cls: type[Self]) -> bool:
        ...


class Namespace(Protocol):
    Float64: DType
    Float32: DType
    Int64: DType
    Int32: DType
    Int16: DType
    Int8: DType
    UInt64: DType
    UInt32: DType
    UInt16: DType
    UInt8: DType
    Bool: DType
    String: DType

    # --- selection ---
    def col(self, *names: str | Iterable[str]) -> Expr:
        ...

    def all(self) -> Expr:
        ...

    # --- reduction ---
    def sum(self, *columns: str) -> Expr:
        ...

    def mean(self, *columns: str) -> Expr:
        ...

    def max(self, *columns: str) -> Expr:
        ...

    def min(self, *columns: str) -> Expr:
        ...

    def len(self) -> Expr:
        ...

    # --- horizontal ---
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

    def cast(self, dtype: DType) -> Self:
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

    def zip_with(self, mask: Self, other: Self) -> Self:
        ...

    def sample(self, n: int, fraction: float, *, with_replacement: bool) -> Series:
        ...

    def to_numpy(self) -> Any:
        ...

    def to_pandas(self) -> Any:
        ...


class DataFrame(Protocol):
    # --- properties ---
    @property
    def columns(self) -> list[str]:
        ...

    @property
    def schema(self) -> dict[str, DType]:
        ...

    @property
    def shape(self) -> tuple[int, int]:
        ...

    # --- reshape ---
    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        ...

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        ...

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        ...

    def rename(self, mapping: dict[str, str]) -> Self:
        ...

    # --- transform ---
    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        ...

    # --- convert ---
    def lazy(self) -> LazyFrame:
        ...

    def to_numpy(self) -> Any:
        ...

    def to_pandas(self) -> Any:
        ...

    def to_dict(self, *, as_series: bool = True) -> dict[str, Any]:
        ...

    # --- actions ---
    def join(
        self,
        other: Self,
        *,
        how: Literal["inner"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> Self:
        ...

    def group_by(self, *keys: str | Iterable[str]) -> GroupBy:
        ...

    # --- partial reduction ---
    def head(self, n: int) -> Self:
        ...

    def unique(self, subset: list[str]) -> Self:
        ...

    # --- public, non-Polars ---
    def to_native(self) -> Any:
        ...


class LazyFrame(Protocol):
    # --- properties ---
    @property
    def columns(self) -> list[str]:
        ...

    @property
    def schema(self) -> dict[str, DType]:
        ...

    # --- reshape ---
    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        ...

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        ...

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        ...

    # --- transform ---
    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        ...

    # --- convert ---
    def collect(self) -> DataFrame:
        ...

    # --- actions ---
    def group_by(self, *keys: str | Iterable[str]) -> LazyGroupBy:
        ...

    def join(
        self,
        other: Self,
        *,
        how: Literal["left", "inner", "outer"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> Self:
        ...

    # --- partial reduction ---
    def head(self, n: int) -> Self:
        ...

    def unique(self, subset: list[str]) -> Self:
        ...

    def rename(self, mapping: dict[str, str]) -> Self:
        ...

    # --- lazy-only ---
    def cache(self) -> Self:
        ...

    # --- public, non-Polars ---
    def to_native(self) -> Any:
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
