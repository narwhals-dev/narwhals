from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Sequence

import polars as pl

from narwhals.pandas_like.utils import isinstance_or_issubclass
from narwhals.spec import DataFrame as DataFrameProtocol
from narwhals.spec import DType as DTypeProtocol
from narwhals.spec import Expr as ExprProtocol
from narwhals.spec import ExprStringNamespace as ExprStringNamespaceProtocol
from narwhals.spec import GroupBy as GroupByProtocol
from narwhals.spec import Namespace as NamespaceProtocol
from narwhals.spec import Series as SeriesProtocol
from narwhals.utils import flatten_into_expr

if TYPE_CHECKING:
    from polars.type_aliases import PolarsDataType
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
        return self.__class__(self._expr.cast(reverse_translate_dtype(dtype)))

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
        return reverse_translate_dtype(cls).is_numeric()  # type: ignore[no-any-return]


class NumericType(DType):
    ...


class TemporalType(DType):
    ...


class Int64(NumericType):
    ...


class Int32(NumericType):
    ...


class Int16(NumericType):
    ...


class Int8(NumericType):
    ...


class UInt64(NumericType):
    ...


class UInt32(NumericType):
    ...


class UInt16(NumericType):
    ...


class UInt8(NumericType):
    ...


class Float64(NumericType):
    ...


class Float32(NumericType):
    ...


class String(DType):
    ...


class Boolean(DType):
    ...


class Datetime(TemporalType):
    ...


class Date(TemporalType):
    ...


class Namespace(NamespaceProtocol):
    Float64 = Float64
    Float32 = Float32
    Int64 = Int64
    Int32 = Int32
    Int16 = Int16
    Int8 = Int8
    UInt64 = UInt64
    UInt32 = UInt32
    UInt16 = UInt16
    UInt8 = UInt8
    Boolean = Boolean
    String = String

    def Series(self, name: str, data: list[Any]) -> Series:  # noqa: N802
        import polars as pl

        from narwhals.polars import Series

        return Series(pl.Series(name=name, values=data))

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
        how: str = "vertical",
    ) -> DataFrame:
        if how == "horizontal":
            # TODO: is_eager / is_lazy not really correct here
            return DataFrame(
                pl.concat([extract_native(v) for v in items], how="horizontal"),
                is_eager=True,
                is_lazy=False,
            )
        raise NotImplementedError


class Series(SeriesProtocol):
    def __init__(self, series: pl.Series) -> None:
        self._series = series

    def alias(self, name: str) -> Self:
        return self.__class__(self._series.alias(name))

    def to_native(self) -> Any:
        return self._series

    @property
    def name(self) -> str:
        return self._series.name

    @property
    def dtype(self) -> DType:
        return translate_dtype(self._series.dtype)  # type: ignore[no-any-return]

    @property
    def shape(self) -> tuple[int]:
        return self._series.shape

    def rename(self, name: str) -> Self:
        return self.__class__(self._series.rename(name))

    def cast(
        self,
        dtype: DType,  # type: ignore[override]
    ) -> Self:
        return self.__class__(self._series.cast(reverse_translate_dtype(dtype)))

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

    def __getitem__(self, column_name: str) -> Series:
        if not self._is_eager:
            raise RuntimeError(
                "DataFrame.shape can only be called if frame was instantiated with `is_eager=True`"
            )
        assert isinstance(self._dataframe, pl.DataFrame)
        return Series(
            self._dataframe[column_name],
        )

    # --- properties ---
    @property
    def columns(self) -> list[str]:
        return self._dataframe.columns

    @property
    def schema(self) -> dict[str, DTypeProtocol]:
        return {
            col: translate_dtype(dtype) for col, dtype in self._dataframe.schema.items()
        }

    @property
    def shape(self) -> tuple[int, int]:
        if not self._is_eager:
            raise RuntimeError(
                "DataFrame.shape can only be called if frame was instantiated with `is_eager=True`"
            )
        assert isinstance(self._dataframe, pl.DataFrame)
        return self._dataframe.shape

    def iter_columns(self) -> Iterable[Series]:
        if not self._is_eager:
            raise RuntimeError(
                "DataFrame.iter_columns can only be called if frame was instantiated with `is_eager=True`"
            )
        assert isinstance(self._dataframe, pl.DataFrame)
        return (Series(self._dataframe[col]) for col in self.columns)

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


def reverse_translate_dtype(dtype: DType | type[DType]) -> Any:
    if isinstance_or_issubclass(dtype, Float64):
        return pl.Float64
    if isinstance_or_issubclass(dtype, Float32):
        return pl.Float32
    if isinstance_or_issubclass(dtype, Int64):
        return pl.Int64
    if isinstance_or_issubclass(dtype, Int32):
        return pl.Int32
    if isinstance_or_issubclass(dtype, Int16):
        return pl.Int16
    if isinstance_or_issubclass(dtype, UInt8):
        return pl.UInt8
    if isinstance_or_issubclass(dtype, UInt64):
        return pl.UInt64
    if isinstance_or_issubclass(dtype, UInt32):
        return pl.UInt32
    if isinstance_or_issubclass(dtype, UInt16):
        return pl.UInt16
    if isinstance_or_issubclass(dtype, UInt8):
        return pl.UInt8
    if isinstance_or_issubclass(dtype, String):
        return pl.String
    if isinstance_or_issubclass(dtype, Boolean):
        return pl.Boolean
    if isinstance_or_issubclass(dtype, Datetime):
        return pl.Datetime
    if isinstance_or_issubclass(dtype, Date):
        return pl.Date
    msg = f"Unknown dtype: {dtype}"
    raise TypeError(msg)


def translate_dtype(dtype: PolarsDataType) -> Any:
    if dtype == pl.Float64:
        return Float64
    if dtype == pl.Float32:
        return Float32
    if dtype == pl.Int64:
        return Int64
    if dtype == pl.Int32:
        return Int32
    if dtype == pl.Int16:
        return Int16
    if dtype == pl.UInt8:
        return UInt8
    if dtype == pl.UInt64:
        return UInt64
    if dtype == pl.UInt32:
        return UInt32
    if dtype == pl.UInt16:
        return UInt16
    if dtype == pl.UInt8:
        return UInt8
    if dtype == pl.String:
        return String
    if dtype == pl.Boolean:
        return Boolean
    if dtype == pl.Datetime:
        return Datetime
    if dtype == pl.Date:
        return Date
    msg = f"Unknown dtype: {dtype}"
    raise TypeError(msg)
