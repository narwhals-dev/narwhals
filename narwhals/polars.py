from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import TypeVar

import polars as pl

from narwhals.spec import DataFrame as DataFrameProtocol
from narwhals.spec import DType as DTypeProtocol
from narwhals.spec import Expr as ExprProtocol
from narwhals.spec import ExprStringNamespace as ExprStringNamespaceProtocol
from narwhals.spec import GroupBy as GroupByProtocol
from narwhals.spec import LazyFrame as LazyFrameProtocol
from narwhals.spec import LazyGroupBy as LazyGroupByProtocol
from narwhals.spec import Namespace as NamespaceProtocol
from narwhals.spec import Series as SeriesProtocol
from narwhals.utils import flatten_into_expr

if TYPE_CHECKING:
    import polars as pl
    from typing_extensions import Self


def extract_native(object: Any) -> Any:
    if isinstance(object, Expr):
        return object._expr
    if isinstance(object, DType):
        return object._dtype
    if isinstance(object, DataFrame):
        return object._dataframe
    if isinstance(object, LazyFrame):
        return object._dataframe
    if isinstance(object, Series):
        return object._series
    return object


class Expr(ExprProtocol):
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    # --- convert ---
    def alias(self, name: str) -> Self:
        return self.__class__(self._expr.alias(name))

    def cast(self, dtype: DType) -> Self:
        self.__class__(self._expr.cast(extract_native(dtype)))

    # --- binary ---
    def __eq__(self, other: object) -> Expr:  # type: ignore[override]
        return self.__class__(self._expr.__eq__(extract_native(other)))

    def __and__(self, other: Any) -> Expr:
        return self.__class__(self._expr.__and__(extract_native(other)))

    def __or__(self, other: Any) -> Expr:
        return self.__class__(self._expr.__or__(extract_native(other)))

    def __add__(self, other: Any) -> Expr:
        return self.__class__(self._expr.__add__(extract_native(other)))

    def __radd__(self, other: Any) -> Expr:
        return self.__class__(self._expr.__radd__(extract_native(other)))

    def __sub__(self, other: Any) -> Expr:
        return self.__class__(self._expr.__sub__(extract_native(other)))

    def __rsub__(self, other: Any) -> Expr:
        return self.__class__(self._expr.__rsub__(extract_native(other)))

    def __mul__(self, other: Any) -> Expr:
        return self.__class__(self._expr.__mul__(extract_native(other)))

    def __rmul__(self, other: Any) -> Expr:
        return self.__class__(self._expr.__rmul__(extract_native(other)))

    def __le__(self, other: Any) -> Expr:
        return self.__class__(self._expr.__le__(extract_native(other)))

    def __lt__(self, other: Any) -> Expr:
        return self.__class__(self._expr.__lt__(extract_native(other)))

    def __gt__(self, other: Any) -> Expr:
        return self.__class__(self._expr.__gt__(extract_native(other)))

    def __ge__(self, other: Any) -> Expr:
        return self.__class__(self._expr.__ge__(extract_native(other)))

    # --- unary ---
    def mean(self) -> Expr:
        return self.__class__(self._expr.mean())

    def sum(self) -> Expr:
        return self.__class__(self._expr.sum())

    def min(self) -> Expr:
        return self.__class__(self._expr.min())

    def max(self) -> Expr:
        return self.__class__(self._expr.max())

    def n_unique(self) -> Expr:
        return self.__class__(self._expr.n_unique())

    # --- transform ---
    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Expr:
        return self.__class__(self._expr.is_between(lower_bound, upper_bound, closed))

    def is_in(self, other: Any) -> Expr:
        return self.__class__(self._expr.is_in(other))

    def is_null(self) -> Expr:
        return self.__class__(self._expr.is_null())

    # --- partial reduction ---
    def drop_nulls(self) -> Expr:
        return self.__class__(self._expr.drop_nulls())

    def sample(self, n: int, fraction: float, *, with_replacement: bool) -> Expr:
        return self.__class__(self._expr.sample(n, fraction, with_replacement))

    # --- namespaces ---
    @property
    def str(self) -> ExprStringNamespace:
        return ExprStringNamespace(self._expr.str)


class ExprStringNamespace(ExprStringNamespaceProtocol):
    def __init__(self, expr: Expr) -> None:
        self._expr = expr

    def ends_with(self, suffix: str) -> Expr:
        self.__class__(self._expr.ends_with(suffix))


class DType(DTypeProtocol):
    def __init__(self, dtype):
        self._dtype = dtype

    @classmethod
    def is_numeric(cls: type[Self]) -> bool:
        return self._dtype.is_numeric()


class Namespace(NamespaceProtocol):
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
        import polars as pl

        return Expr(pl.col(*names))

    def all(self) -> Expr:
        return Expr(pl.all())

    # --- reduction ---
    def sum(self, *columns: str) -> Expr:
        return Expr(pl.sum(*columns))

    def mean(self, *columns: str) -> Expr:
        return Expr(pl.mean(*columns))

    def max(self, *columns: str) -> Expr:
        return Expr(pl.max(*columns))

    def min(self, *columns: str) -> Expr:
        return Expr(pl.min(*columns))

    def len(self) -> Expr:
        return Expr(pl.len())

    # --- horizontal ---
    def all_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
        return Expr(pl.all_horizontal(*[extract_native(v) for v in exprs]))

    def any_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
        return Expr(pl.any_horizontal(*[extract_native(v) for v in exprs]))

    def sum_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
        return Expr(pl.sum_horizontal(*[extract_native(v) for v in exprs]))

    def concat(self, items: Iterable[AnyDataFrame], *, how: str) -> AnyDataFrame:
        # bit harder, do this later
        ...


class Series(SeriesProtocol):
    def __init__(self, series: pl.Series):
        self._series = series

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

    def zip_with(self, mask: Series, other: Series) -> Series:
        ...

    def sample(self, n: int, fraction: float, *, with_replacement: bool) -> Series:
        ...

    def to_numpy(self) -> Any:
        ...

    def to_pandas(self) -> Any:
        ...


class DataFrame(DataFrameProtocol):
    def __init__(self, df: pl.DataFrame):
        self._dataframe = df

    # --- properties ---
    @property
    def columns(self) -> list[str]:
        return self._dataframe.columns

    @property
    def schema(self) -> dict[str, DType]:
        return {key: DType(value) for key, value in self._dataframe.schema.items()}

    @property
    def shape(self) -> tuple[int, int]:
        return self._dataframe.shape

    # --- reshape ---
    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.__class__(
            self._dataframe.with_columns(
                *[extract_native(v) for v in exprs],
                **{key: extract_native(value) for key, value in named_exprs.items()},
            )
        )

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        return self.__class__(
            self._dataframe.filter(*[extract_native(v) for v in predicates])
        )

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.__class__(
            self._dataframe.select(
                *[extract_native(v) for v in exprs],
                **{key: extract_native(value) for key, value in named_exprs.items()},
            )
        )

    def rename(self, mapping: dict[str, str]) -> Self:
        return self.__class__(self._dataframe.rename(mapping))

    # --- transform ---
    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Iterable[bool] = False,
    ) -> Self:
        return self.__class__(self._dataframe.sort(by, *more_by, descending=descending))

    # --- convert ---
    def lazy(self) -> LazyFrame:
        return LazyFrame(self._dataframe.lazy())

    def to_numpy(self) -> Any:
        return self._dataframe.to_numpy()

    def to_pandas(self) -> Any:
        return self._dataframe.to_pandas()

    def to_dict(self, *, as_series: bool = True) -> dict[str, Any]:
        return self._dataframe.to_dict(as_series=as_series)

    # --- actions ---
    def join(
        self,
        other: Self,
        *,
        how: Literal["inner"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> Self:
        return self.__class__(
            self._dataframe.join(
                extract_native(other), how=how, left_on=left_on, right_on=right_on
            )
        )

    def group_by(self, *keys: str | Iterable[str]) -> GroupBy:
        return self.__class__(self._dataframe.group_by(*keys))

    # --- partial reduction ---
    def head(self, n: int) -> Self:
        return self.__class__(self._dataframe.head(n))

    def unique(self, subset: list[str]) -> Self:
        return self.__class__(self._dataframe.unique(subset))

    # --- public, non-Polars ---
    def to_native(self) -> Any:
        return self._dataframe


class LazyFrame(LazyFrameProtocol):
    def __init__(self, df):
        self._dataframe = df

    # --- properties ---
    @property
    def columns(self) -> list[str]:
        return self._dataframe.columns

    @property
    def schema(self) -> dict[str, DType]:
        return {key: DType(value) for key, value in self._dataframe.schema.items()}

    # --- reshape ---
    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.__class__(
            self._dataframe.with_columns(
                *[extract_native(v) for v in exprs],
                **{key: extract_native(value) for key, value in named_exprs.items()},
            )
        )

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        return self.__class__(
            self._dataframe.filter(*[extract_native(v) for v in predicates])
        )

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.__class__(
            self._dataframe.select(
                *[extract_native(v) for v in exprs],
                **{key: extract_native(value) for key, value in named_exprs.items()},
            )
        )

    # --- transform ---
    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Iterable[bool] = False,
    ) -> Self:
        return __class__(self._dataframe.sort(by, *more_by, descending=descending))

    # --- convert ---
    def collect(self) -> DataFrame:
        return DataFrame(self._dataframe.collect())

    # --- actions ---
    def group_by(self, *keys: str | Iterable[str]) -> LazyGroupBy:
        return LazyGroupBy(self._dataframe.group_by(*keys))

    def join(
        self,
        other: Self,
        *,
        how: Literal["left", "inner", "outer"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> Self:
        return self.__class__(
            self._dataframe.join(
                extract_native(other), how=how, left_on=left_on, right_on=right_on
            )
        )

    # --- partial reduction ---
    def head(self, n: int) -> Self:
        return self.__class__(self._dataframe.head(n))

    def unique(self, subset: list[str]) -> Self:
        return self.__class__(self._dataframe.unique(subset))

    def rename(self, mapping: dict[str, str]) -> Self:
        return self.__class__(self._dataframe.rename(mapping))

    # --- lazy-only ---
    def cache(self) -> Self:
        return self.__class__(self._dataframe.cache())

    # --- public, non-Polars ---
    def to_native(self) -> Any:
        return self._dataframe


class GroupBy(GroupByProtocol):
    def __init__(self, groupby) -> None:
        self._groupby = groupby

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> DataFrame:
        return DataFrame(
            self._groupby.agg(
                *[extract_native(v) for v in aggs],
                **{key: extract_native(value) for key, value in named_aggs.items()},
            )
        )


class LazyGroupBy(LazyGroupByProtocol):
    def __init__(self, groupby):
        self._groupby = groupby

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> LazyFrame:
        return LazyFrame(
            self._groupby.agg(
                *[extract_native(v) for v in flatten_into_expr(*aggs)],
                **{key: extract_native(value) for key, value in named_aggs.items()},
            )
        )


IntoExpr = Expr | str | int | float | Series

AnyDataFrame = TypeVar("AnyDataFrame", DataFrame, LazyFrame)
