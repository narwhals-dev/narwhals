from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, overload

import polars as pl

from narwhals._plan._namespace import namespace
from narwhals._plan._version import into_version
from narwhals._plan.compliant.accessors import SeriesStructNamespace as StructNamespace
from narwhals._plan.compliant.series import CompliantSeries
from narwhals._plan.compliant.typing import SeriesT
from narwhals._plan.polars import compat
from narwhals._plan.polars.namespace import (
    PolarsNamespace as Namespace,
    dtype_from_native,
    dtype_to_native,
    dtype_to_native_fast,
    explode_todo,
)
from narwhals._utils import Implementation, Version, requires
from narwhals.dependencies import is_numpy_array_1d, is_pandas_index

if TYPE_CHECKING:
    import datetime as dt
    import decimal
    from collections.abc import Callable, Iterable

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.polars.dataframe import (
        PolarsDataFrame,
        PolarsDataFrame as DataFrame,
    )
    from narwhals.dtypes import DType, IntegerType
    from narwhals.schema import Schema
    from narwhals.typing import (
        ClosedInterval,
        FillNullStrategy,
        Into1DArray,
        IntoDType,
        NonNestedLiteral,
        PythonLiteral,
        _1DArray,
    )

Incomplete: TypeAlias = Any
Int64 = Version.MAIN.dtypes.Int64()
if compat.MIN_PERIODS_RENAMED_TO_MIN_SAMPLES:
    _MIN_SAMPLES = "min_samples"
else:
    _MIN_SAMPLES = "min_periods"


def min_samples_periods(min_samples: int, /, **kwds: Any) -> dict[str, Any]:
    return {_MIN_SAMPLES: min_samples, **kwds}


# TODO @dangotbanned: Remove the `getattr` indirection and just bind the normal way
# Needing an inner function to get the original idea working loses the benefit of skipping
# the creation of a new function
def _make_bin_op(name: str, /) -> Callable[[SeriesT], Callable[[Any], SeriesT]]:
    method_native = getattr(pl.Series, name)

    def f(self: SeriesT, /) -> Callable[[Any], SeriesT]:
        def inner(other: Any, /) -> SeriesT:
            other = other.native if isinstance(other, type(self)) else other
            result = method_native(self.native, other)
            return self.from_native(result)

        return inner

    return f


class bin_op(Generic[SeriesT]):  # noqa: N801
    """Descriptor adding a lazy proxy for binary Series operations.

    Note:
        - `pyright` is fine with a `TypeVar` default of `PolarsSeries`
        - `mypy` requires annotating as `bin_op[Self]`
    """

    __slots__ = ("__name__", "_method_native", "_name_owner")

    def __init__(self) -> None:
        self._method_native: Callable[[SeriesT], Callable[[Any], SeriesT]] | None = None
        """Generated *iff* the method was ever used.

        After the first call, the same wrapper function is reused for all instances.
        """

    def __set_name__(self, owner: type[SeriesT], name: str) -> None:
        self._name_owner: str = owner.__name__
        self.__name__: str = name

    def __repr__(self) -> str:
        return f"bin_op<{self._name_owner}.{self.__name__}>"

    @overload
    def __get__(self, instance: SeriesT, owner: Any, /) -> Callable[[Any], SeriesT]: ...
    @overload
    def __get__(self, instance: None, owner: type[SeriesT], /) -> Self: ...
    def __get__(
        self, instance: SeriesT | None, owner: type[SeriesT] | None, /
    ) -> Self | Callable[[Any], SeriesT]:
        if instance is None:
            return self
        if self._method_native is None:
            self._method_native = _make_bin_op(self.__name__)
        return self._method_native(instance)

    def __call__(self, instance: SeriesT, other: Any, /) -> SeriesT:
        raise NotImplementedError


class PolarsSeries(CompliantSeries[pl.Series]):
    __slots__ = ("_native",)
    implementation = Implementation.POLARS
    _native: pl.Series
    version: ClassVar[Version] = Version.MAIN

    # NOTE: Aliases to integrate with `@requires.backend_version`
    _backend_version = compat.BACKEND_VERSION
    _implementation = implementation

    @property
    def native(self) -> pl.Series:
        return self._native

    @property
    def name(self) -> str:
        return self.native.name

    @property
    def dtype(self) -> DType:
        return dtype_from_native(self.native.dtype, self.version)

    def __narwhals_namespace__(self) -> Namespace:
        return Namespace()

    @classmethod
    def from_iterable(
        cls, data: Iterable[Any], *, name: str = "", dtype: IntoDType | None = None
    ) -> Self:
        dtype_pl = dtype_to_native(dtype, cls.version)
        values: Incomplete = data
        if compat.SERIES_RESPECTS_DTYPE:
            native = pl.Series(name, values, dtype=dtype_pl)
        else:  # pragma: no cover
            if (not compat.SERIES_ACCEPTS_PD_INDEX) and is_pandas_index(values):
                values = values.to_series()
            native = pl.Series(name, values)
            if dtype_pl:
                native = native.cast(dtype_pl)
        return cls.from_native(native)

    @classmethod
    def from_native(cls, native: pl.Series, name: str = "", /) -> Self:
        obj = cls.__new__(cls)
        obj._native = native if not name else native.alias(name)
        return obj

    @classmethod
    def from_numpy(cls, data: Into1DArray, name: str = "", /) -> Self:
        native = pl.Series(data if is_numpy_array_1d(data) else [data])
        return cls.from_native(native, name)

    def _with_native(self, native: pl.Series) -> Self:
        return self.from_native(native)

    def to_list(self) -> list[Any]:
        return self.native.to_list()

    def to_polars(self) -> pl.Series:
        return self.native

    def cast(self, dtype: IntoDType) -> Self:
        result = self.native.cast(dtype_to_native(dtype, self.version))
        return self._with_native(result)

    def has_nulls(self) -> bool:
        return self.native.has_nulls()

    def is_in(self, other: Self) -> Self:
        return self._with_native(self.native.is_in(other.native))

    # NOTE: Needs compat
    # https://github.com/narwhals-dev/narwhals/blob/c207fc096263ce174470240748e0c568f38f93e2/narwhals/_polars/series.py#L364-L372
    def is_nan(self) -> Self:
        if compat.IS_NAN_NUMERIC_PROPAGATES_NULLS:
            return self._with_native(self.native.is_nan())
        msg = "TODO @dangotbanned: `is_nan` backcompat\nSee https://github.com/narwhals-dev/narwhals/pull/1625#issuecomment-2565591385"
        raise NotImplementedError(msg)

    def is_null(self) -> Self:
        return self._with_native(self.native.is_null())

    def is_not_nan(self) -> Self:
        return self._with_native(self.native.is_not_nan())

    def is_not_null(self) -> Self:
        return self._with_native(self.native.is_not_null())

    def to_frame(self) -> DataFrame:
        df = self.native.to_frame()
        return namespace(self)._dataframe.from_native(df)

    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _1DArray:
        return self.__array__(dtype, copy=copy)

    def __array__(self, dtype: Any, *, copy: bool | None) -> _1DArray:
        method = self.native.__array__
        return method(dtype, copy) if compat.DUNDER_ARRAY_SUPPORTS_COPY else method(dtype)

    def first(self) -> PythonLiteral | Incomplete:
        if compat.SERIES_HAS_FIRST_LAST:
            return self.native.first()
        return None if self.is_empty() else self.native.item(0)

    def last(self) -> PythonLiteral | Incomplete:
        if compat.SERIES_HAS_FIRST_LAST:
            return self.native.last()
        return None if self.is_empty() else self.native.item(-1)

    def fill_nan(self, value: float | Self | None) -> Self:
        fill_value = pl.lit(value.native) if isinstance(value, PolarsSeries) else value
        return self._with_native(self.native.fill_nan(fill_value))

    def fill_null(self, value: NonNestedLiteral | Self) -> Self:
        fill_value = value.native if isinstance(value, PolarsSeries) else value
        return self._with_native(self.native.fill_null(fill_value))

    def fill_null_with_strategy(
        self, strategy: FillNullStrategy, limit: int | None = None
    ) -> Self:
        return self._with_native(self.native.fill_null(strategy=strategy, limit=limit))

    def explode(self, *, empty_as_null: bool = True, keep_nulls: bool = True) -> Self:
        explode_todo(empty_as_null=empty_as_null, keep_nulls=keep_nulls)
        return self._with_native(self.native.explode())

    def __invert__(self) -> Self:
        return self._with_native(self.native.__invert__())

    __add__ = bin_op["Self"]()
    __and__ = bin_op["Self"]()
    __eq__ = bin_op["Self"]()
    __floordiv__ = bin_op["Self"]()
    __ge__ = bin_op["Self"]()
    __gt__ = bin_op["Self"]()
    __le__ = bin_op["Self"]()
    __lt__ = bin_op["Self"]()
    __mod__ = bin_op["Self"]()
    __mul__ = bin_op["Self"]()
    __ne__ = bin_op["Self"]()
    __or__ = bin_op["Self"]()
    __pow__ = bin_op["Self"]()
    __radd__ = bin_op["Self"]()
    __rand__ = bin_op["Self"]()
    __rmod__ = bin_op["Self"]()
    __rmul__ = bin_op["Self"]()
    __ror__ = bin_op["Self"]()
    __rsub__ = bin_op["Self"]()
    __rtruediv__ = bin_op["Self"]()
    __rxor__ = bin_op["Self"]()
    __sub__ = bin_op["Self"]()
    __truediv__ = bin_op["Self"]()
    __xor__ = bin_op["Self"]()

    def __rfloordiv__(self, other: Any) -> PolarsSeries:
        other = other.native if isinstance(other, type(self)) else other
        if compat.SERIES_RFLOORDIV_HANDLES_ZERO:
            return self._with_native(other // self.native)
        expr = pl.col(self.name)
        return self._with_native(
            self.native.to_frame()
            .select(pl.when(expr != 0).then(other // expr).alias(self.name))
            .to_series()
        )

    def __rpow__(self, other: float | Self) -> Self:
        other_ = other.native if isinstance(other, PolarsSeries) else other
        result = other_**self.native
        if not compat.SERIES_RPOW_PRESERVES_NAME:
            result = result.alias(self.native.name)
        return self._with_native(result)

    def all(self) -> bool:
        return self.native.all()

    def any(self) -> bool:
        return self.native.any()

    def count(self) -> int:
        return self.native.count()

    def cum_count(self, *, reverse: bool = False) -> Self:
        return self._with_native(self.native.cum_count(reverse=reverse))

    def cum_max(self, *, reverse: bool = False) -> Self:
        return self._with_native(self.native.cum_max(reverse=reverse))

    def cum_min(self, *, reverse: bool = False) -> Self:
        return self._with_native(self.native.cum_min(reverse=reverse))

    def cum_prod(self, *, reverse: bool = False) -> Self:
        return self._with_native(self.native.cum_prod(reverse=reverse))

    def cum_sum(self, *, reverse: bool = False) -> Self:
        return self._with_native(self.native.cum_sum(reverse=reverse))

    def diff(self, n: int = 1) -> Self:
        return self._with_native(self.native.diff(n))

    def drop_nans(self) -> Self:
        return self._with_native(self.native.drop_nans())

    def drop_nulls(self) -> Self:
        return self._with_native(self.native.drop_nulls())

    def gather(self, indices: Self) -> Self:
        return self._with_native(self.native.gather(indices.native))

    def gather_every(self, n: int, offset: int = 0) -> Self:
        return self._with_native(self.native.gather_every(n, offset))

    def null_count(self) -> int:
        return self.native.null_count()

    def rolling_mean(
        self, window_size: int, *, min_samples: int, center: bool = False
    ) -> Self:
        kwds = min_samples_periods(min_samples, window_size=window_size, center=center)
        return self._with_native(self.native.rolling_mean(**kwds))

    def rolling_sum(
        self, window_size: int, *, min_samples: int, center: bool = False
    ) -> Self:
        kwds = min_samples_periods(min_samples, window_size=window_size, center=center)
        return self._with_native(self.native.rolling_sum(**kwds))

    @requires.backend_version((1,))
    def rolling_std(
        self, window_size: int, *, min_samples: int, center: bool = False, ddof: int = 1
    ) -> Self:
        kwds = min_samples_periods(
            min_samples, window_size=window_size, center=center, ddof=ddof
        )
        return self._with_native(self.native.rolling_std(**kwds))

    @requires.backend_version((1,))
    def rolling_var(
        self, window_size: int, *, min_samples: int, center: bool = False, ddof: int = 1
    ) -> Self:
        kwds = min_samples_periods(
            min_samples, window_size=window_size, center=center, ddof=ddof
        )
        return self._with_native(self.native.rolling_var(**kwds))

    def sample_n(
        self, n: int = 1, *, with_replacement: bool = False, seed: int | None = None
    ) -> Self:
        return self._with_native(
            self.native.sample(n, with_replacement=with_replacement, seed=seed)
        )

    def scatter(self, indices: Self, values: Self) -> Self:
        return self._with_native(
            self.native.clone().scatter(indices.native, values.native)
        )

    def shift(self, n: int, *, fill_value: NonNestedLiteral = None) -> Self:
        return self._with_native(self.native.shift(n, fill_value=fill_value))

    def slice(self, offset: int, length: int | None = None) -> Self:
        return self._with_native(self.native.slice(offset, length))

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        if compat.SERIES_SORT_SUPPORTS_NULLS_LAST:
            result = self.native.sort(descending=descending, nulls_last=nulls_last)
        elif not (nulls_last and self.has_nulls()):
            result = self.native.sort(descending=descending)
        else:
            result = (
                self.native.to_frame()
                .sort(self.name, descending=descending, nulls_last=nulls_last)
                .to_series()
            )
        return self._with_native(result)

    def sum(self) -> float | decimal.Decimal:
        return self.native.sum()

    def unique(self, *, maintain_order: bool = False) -> Self:
        return self._with_native(self.native.unique(maintain_order=maintain_order))

    def zip_with(self, mask: Self, other: Self) -> Self:
        return self._with_native(self.native.zip_with(mask.native, other.native))

    @classmethod
    def concat(cls, series: Iterable[Self]) -> Self:
        return cls.from_native(pl.concat(ser.native for ser in series))

    @classmethod
    def date_range(
        cls,
        start: dt.date,
        end: dt.date,
        interval: int = 1,
        *,
        closed: ClosedInterval = "both",
        name: str = "literal",
    ) -> Self:
        native = pl.date_range(start, end, f"{interval}d", closed=closed, eager=True)
        return cls.from_native(native, name)

    @classmethod
    def int_range(
        cls,
        start: int,
        end: int,
        step: int = 1,
        *,
        dtype: IntegerType = Int64,
        name: str = "literal",
    ) -> Self:
        dtype_ = dtype_to_native_fast(dtype)
        native = pl.int_range(start, end, step, dtype=dtype_, eager=True)
        return cls.from_native(native, name)

    @classmethod
    def linear_space(
        cls,
        start: float,
        end: float,
        num_samples: int,
        *,
        closed: ClosedInterval = "both",
        name: str = "literal",
    ) -> Self:
        native = pl.linear_space(start, end, num_samples, closed=closed, eager=True)
        return cls.from_native(native, name)

    @property
    def struct(self) -> SeriesStructNamespace:
        return SeriesStructNamespace(self)


class SeriesStructNamespace(StructNamespace["DataFrame", PolarsSeries]):
    __slots__ = ("_compliant",)

    def __init__(self, compliant: PolarsSeries, /) -> None:
        self._compliant: PolarsSeries = compliant

    @property
    def compliant(self) -> PolarsSeries:
        return self._compliant

    @property
    def native(self) -> pl.Series:
        return self.compliant.native

    def field(self, name: str) -> PolarsSeries:
        return self.compliant._with_native(self.native.struct.field(name))

    def unnest(self) -> PolarsDataFrame:
        df = self.native.struct.unnest()
        return self.compliant.__narwhals_namespace__()._dataframe.from_native(df)

    @property
    def schema(self) -> Schema:
        return into_version(self.version).schema.from_polars(self.native.struct.schema)


PolarsSeries()
