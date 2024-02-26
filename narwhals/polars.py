from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Sequence

import polars as pl

from narwhals.spec import DataFrame as DataFrameProtocol
from narwhals.spec import DType as DTypeProtocol
from narwhals.spec import Expr as ExprProtocol
from narwhals.spec import ExprStringNamespace as ExprStringNamespaceProtocol
from narwhals.spec import GroupBy as GroupByProtocol
from narwhals.spec import Namespace as NamespaceProtocol
from narwhals.spec import Series as SeriesProtocol
from narwhals.utils import flatten_into_expr

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.spec import IntoExpr


def extract_native(obj: Any) -> Any:
    if isinstance(obj, Expr):
        return obj._expr
    if isinstance(obj, DType):
        return obj._dtype
    if isinstance(obj, DataFrame):
        return obj._dataframe
    if isinstance(obj, Series):
        return obj._series
    return obj


class Expr(ExprProtocol):
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    # --- convert ---
    def alias(self, name: str) -> Self:
        return self.__class__(self._expr.alias(name))

    def cast(
        self,
        dtype: DType,  # type: ignore[override]
    ) -> Self:
        return self.__class__(self._expr.cast(extract_native(dtype)))

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
        return self.__class__(self._expr.is_between(lower_bound, upper_bound, closed))  # type: ignore[arg-type]

    def is_in(self, other: Any) -> Expr:
        return self.__class__(self._expr.is_in(other))

    def is_null(self) -> Expr:
        return self.__class__(self._expr.is_null())

    # --- partial reduction ---
    def drop_nulls(self) -> Expr:
        return self.__class__(self._expr.drop_nulls())

    def sample(self, n: int, fraction: float, *, with_replacement: bool) -> Expr:
        return self.__class__(
            self._expr.sample(n, fraction=fraction, with_replacement=with_replacement)
        )

    # --- namespaces ---
    @property
    def str(self) -> ExprStringNamespace:
        return ExprStringNamespace(self._expr.str)


class ExprStringNamespace(ExprStringNamespaceProtocol):
    def __init__(self, expr: Any) -> None:
        self._expr = expr

    def ends_with(self, suffix: str) -> Expr:
        return Expr(self._expr.str.ends_with(suffix))


class DType(DTypeProtocol):
    def __init__(self, dtype: Any) -> None:
        self._dtype = dtype

    @classmethod
    def is_numeric(cls: type[Self]) -> bool:
        return cls._dtype.is_numeric()  # type: ignore[no-any-return]


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
        return Expr(pl.col(*names))  # type: ignore[arg-type]

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

    def concat(
        self,
        items: Iterable[DataFrame],  # type: ignore[override]
        *,
        how: str,
    ) -> DataFrame:
        # bit harder, do this later
        raise NotImplementedError


class Series(SeriesProtocol):
    def __init__(self, series: pl.Series) -> None:
        self._series = series

    def alias(self, name: str) -> Self:
        return self.__class__(self._series.alias(name))

    @property
    def name(self) -> str:
        return self._series.name

    def cast(
        self,
        dtype: DType,  # type: ignore[override]
    ) -> Self:
        return self.__class__(self._series.cast(extract_native(dtype)))

    def item(self) -> Any:
        return self._series.item()

    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Series:
        return self.__class__(self._series.is_between(lower_bound, upper_bound, closed))  # type: ignore[arg-type]

    def is_in(self, other: Any) -> Series:
        return self.__class__(self._series.is_in(other))

    def is_null(self) -> Series:
        return self.__class__(self._series.is_null())

    def drop_nulls(self) -> Series:
        return self.__class__(self._series.drop_nulls())

    def n_unique(self) -> int:
        return self._series.n_unique()

    def zip_with(self, mask: Self, other: Self) -> Self:
        return self.__class__(
            self._series.zip_with(extract_native(mask), extract_native(other))
        )

    def sample(self, n: int, fraction: float, *, with_replacement: bool) -> Series:
        return self.__class__(
            self._series.sample(n, fraction=fraction, with_replacement=with_replacement)
        )

    def to_numpy(self) -> Any:
        return self._series.to_numpy()

    def to_pandas(self) -> Any:
        return self._series.to_pandas()


class DataFrame(DataFrameProtocol):
    def __init__(
        self, df: pl.DataFrame | pl.LazyFrame, *, is_eager: bool, is_lazy: bool
    ) -> None:
        self._dataframe = df
        self._is_eager = is_eager
        self._is_lazy = is_lazy

    def _from_dataframe(self, df: pl.DataFrame | pl.LazyFrame) -> Self:
        # construct, preserving properties
        return self.__class__(df, is_eager=self._is_eager, is_lazy=self._is_lazy)

    # --- properties ---
    @property
    def columns(self) -> list[str]:
        return self._dataframe.columns

    @property
    def schema(self) -> dict[str, DTypeProtocol]:
        return {key: DType(value) for key, value in self._dataframe.schema.items()}

    @property
    def shape(self) -> tuple[int, int]:
        if not self._is_eager:
            raise RuntimeError(
                "DataFrame.shape can only be called if frame was instantiated with `is_eager=True`"
            )
        assert isinstance(self._dataframe, pl.DataFrame)
        return self._dataframe.shape

    # --- reshape ---
    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self._from_dataframe(
            self._dataframe.with_columns(
                *[extract_native(v) for v in exprs],
                **{key: extract_native(value) for key, value in named_exprs.items()},
            )
        )

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        return self._from_dataframe(
            self._dataframe.filter(*[extract_native(v) for v in predicates])
        )

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self._from_dataframe(
            self._dataframe.select(
                *[extract_native(v) for v in exprs],
                **{key: extract_native(value) for key, value in named_exprs.items()},
            )
        )

    def rename(self, mapping: dict[str, str]) -> Self:
        return self._from_dataframe(self._dataframe.rename(mapping))

    # --- transform ---
    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        return self._from_dataframe(
            self._dataframe.sort(by, *more_by, descending=descending)
        )

    # --- convert ---
    def lazy(self) -> Self:
        return self.__class__(self._dataframe.lazy(), is_eager=False, is_lazy=True)

    def collect(self) -> Self:
        if not self._is_lazy:
            raise RuntimeError(
                "DataFrame.collect can only be called if frame was instantiated with `is_lazy=True`"
            )
        assert isinstance(self._dataframe, pl.LazyFrame)
        return self.__class__(self._dataframe.collect(), is_eager=True, is_lazy=False)

    def cache(self) -> Self:
        if not self._is_lazy:
            raise RuntimeError(
                "DataFrame.cache can only be called if frame was instantiated with `is_lazy=True`"
            )
        assert isinstance(self._dataframe, pl.LazyFrame)
        return self.__class__(self._dataframe.cache(), is_eager=False, is_lazy=True)

    def to_numpy(self) -> Any:
        if not self._is_eager:
            raise RuntimeError(
                "DataFrame.to_numpy can only be called if frame was instantiated with `is_eager=True`"
            )
        assert isinstance(self._dataframe, pl.DataFrame)
        return self._dataframe.to_numpy()

    def to_pandas(self) -> Any:
        if not self._is_eager:
            raise RuntimeError(
                "DataFrame.to_pandas can only be called if frame was instantiated with `is_eager=True`"
            )
        assert isinstance(self._dataframe, pl.DataFrame)
        return self._dataframe.to_pandas()

    def to_dict(self, *, as_series: bool = True) -> dict[str, Any]:
        if not self._is_eager:
            raise RuntimeError(
                "DataFrame.to_dict can only be called if frame was instantiated with `is_eager=True`"
            )
        assert isinstance(self._dataframe, pl.DataFrame)
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
        # todo validate eager/lazy only
        return self._from_dataframe(
            self._dataframe.join(
                extract_native(other), how=how, left_on=left_on, right_on=right_on
            )
        )

    def group_by(self, *keys: str | Iterable[str]) -> GroupBy:
        return GroupBy(
            self._dataframe.group_by(*keys),
            is_eager=self._is_eager,
            is_lazy=self._is_lazy,
        )

    # --- partial reduction ---
    def head(self, n: int) -> Self:
        return self._from_dataframe(self._dataframe.head(n))

    def unique(self, subset: list[str]) -> Self:
        return self._from_dataframe(self._dataframe.unique(subset))

    # --- public, non-Polars ---
    def to_native(self) -> Any:
        return self._dataframe

    @property
    def is_eager(self) -> bool:
        return self._is_eager

    @property
    def is_lazy(self) -> bool:
        return self._is_lazy


class GroupBy(GroupByProtocol):
    def __init__(self, groupby: Any, *, is_eager: bool, is_lazy: bool) -> None:
        self._groupby = groupby
        self._is_eager = is_eager
        self._is_lazy = is_lazy

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> DataFrame:
        return DataFrame(
            self._groupby.agg(
                *[extract_native(v) for v in flatten_into_expr(*aggs)],
                **{key: extract_native(value) for key, value in named_aggs.items()},
            ),
            is_eager=self._is_eager,
            is_lazy=self._is_lazy,
        )
