from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
from typing_extensions import Self

from narwhals._plan._namespace import namespace
from narwhals._plan._version import into_version
from narwhals._plan.common import todo
from narwhals._plan.compliant.accessors import SeriesStructNamespace as StructNamespace
from narwhals._plan.compliant.series import CompliantSeries
from narwhals._plan.polars.namespace import (
    PolarsNamespace as Namespace,
    dtype_from_native,
    dtype_to_native,
    explode_todo,
)
from narwhals._polars.utils import (
    BACKEND_VERSION,
    SERIES_ACCEPTS_PD_INDEX,
    SERIES_RESPECTS_DTYPE,
)
from narwhals._utils import Implementation, Version
from narwhals.dependencies import is_numpy_array_1d, is_pandas_index

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.polars.dataframe import (
        PolarsDataFrame,
        PolarsDataFrame as DataFrame,
    )
    from narwhals.dtypes import DType
    from narwhals.schema import Schema
    from narwhals.typing import (
        FillNullStrategy,
        Into1DArray,
        IntoDType,
        NonNestedLiteral,
        PythonLiteral,
        _1DArray,
    )

Incomplete: TypeAlias = Any

SERIES_HAS_FIRST_LAST = BACKEND_VERSION >= (1, 10)
"""https://github.com/pola-rs/polars/pull/19093"""

IS_NAN_ANY_NUMERIC = BACKEND_VERSION >= (1, 18)
"""https://github.com/pola-rs/polars/pull/20386"""

# NOTE: (10-20) already had impls, should detect that during generation
# bug?:
#   __hash__


class PolarsSeries(CompliantSeries[pl.Series]):
    implementation = Implementation.POLARS
    _native: pl.Series
    _version: Version

    @property
    def native(self) -> pl.Series:
        return self._native

    @property
    def name(self) -> str:
        return self.native.name

    @property
    def dtype(self) -> DType:
        return dtype_from_native(self.native.dtype, self._version)

    def __narwhals_namespace__(self) -> Namespace:
        return Namespace(self.version)

    @classmethod
    def from_iterable(
        cls,
        data: Iterable[Any],
        *,
        version: Version,
        name: str = "",
        dtype: IntoDType | None = None,
    ) -> Self:
        dtype_pl = dtype_to_native(dtype, version)
        values: Incomplete = data
        if SERIES_RESPECTS_DTYPE:
            native = pl.Series(name, values, dtype=dtype_pl)
        else:  # pragma: no cover
            if (not SERIES_ACCEPTS_PD_INDEX) and is_pandas_index(values):
                values = values.to_series()
            native = pl.Series(name, values)
            if dtype_pl:
                native = native.cast(dtype_pl)
        return cls.from_native(native, version=version)

    @classmethod
    def from_native(
        cls, native: pl.Series, name: str = "", /, *, version: Version = Version.MAIN
    ) -> Self:
        obj = cls.__new__(cls)
        obj._native = native if not name else native.alias(name)
        obj._version = version
        return obj

    @classmethod
    def from_numpy(
        cls, data: Into1DArray, name: str = "", /, *, version: Version = Version.MAIN
    ) -> Self:
        native = pl.Series(data if is_numpy_array_1d(data) else [data])
        return cls.from_native(native, name, version=version)

    def _with_native(self, native: pl.Series) -> Self:
        return self.from_native(native, version=self.version)

    def to_list(self) -> list[Any]:
        return self.native.to_list()

    def to_polars(self) -> pl.Series:
        return self.native

    def cast(self, dtype: IntoDType) -> Self:
        result = self.native.cast(dtype_to_native(dtype, self._version))
        return self._with_native(result)

    def has_nulls(self) -> bool:
        return self.native.has_nulls()

    def is_in(self, other: Self) -> Self:
        return self._with_native(self.native.is_in(other.native))

    # NOTE: Needs compat
    # https://github.com/narwhals-dev/narwhals/blob/c207fc096263ce174470240748e0c568f38f93e2/narwhals/_polars/series.py#L364-L372
    def is_nan(self) -> Self:
        if IS_NAN_ANY_NUMERIC:
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
        return namespace(self)._dataframe.from_native(df, self.version)

    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _1DArray:
        return self.__array__(dtype, copy=copy)

    def __array__(self, dtype: Any, *, copy: bool | None) -> _1DArray:
        method = self.native.__array__
        if BACKEND_VERSION < (0, 20, 29):
            return method(dtype=dtype)
        return method(dtype=dtype, copy=copy)

    def first(self) -> PythonLiteral | Incomplete:
        if SERIES_HAS_FIRST_LAST:
            return self.native.first()
        return None if self.is_empty() else self.native.item(0)

    def last(self) -> PythonLiteral | Incomplete:
        if SERIES_HAS_FIRST_LAST:
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

    __add__ = todo()
    __and__ = todo()
    __eq__ = todo()
    __floordiv__ = todo()
    __ge__ = todo()
    __gt__ = todo()
    __invert__ = todo()
    __le__ = todo()
    __lt__ = todo()
    __mod__ = todo()
    __mul__ = todo()
    __ne__ = todo()
    __or__ = todo()
    __pow__ = todo()
    __radd__ = todo()
    __rand__ = todo()
    # NOTE: Needs compat
    # https://github.com/narwhals-dev/narwhals/blob/c207fc096263ce174470240748e0c568f38f93e2/narwhals/_polars/series.py#L259-L268
    __rfloordiv__ = todo()
    __rmod__ = todo()
    __rmul__ = todo()
    __ror__ = todo()
    # # NOTE: Needs compat
    # https://github.com/narwhals-dev/narwhals/blob/c207fc096263ce174470240748e0c568f38f93e2/narwhals/_polars/series.py#L357-L362
    __rpow__ = todo()
    __rsub__ = todo()
    __rtruediv__ = todo()
    __rxor__ = todo()
    __sub__ = todo()
    __truediv__ = todo()
    __xor__ = todo()

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

    # NOTE: Needs compat (but check Expr first)
    # https://github.com/narwhals-dev/narwhals/blob/c207fc096263ce174470240748e0c568f38f93e2/narwhals/_polars/series.py#L441-L494
    rolling_mean = todo()
    rolling_std = todo()
    rolling_sum = todo()
    rolling_var = todo()

    def sample_n(
        self, n: int = 1, *, with_replacement: bool = False, seed: int | None = None
    ) -> Self:
        return self._with_native(
            self.native.sample(n, with_replacement=with_replacement, seed=seed)
        )

    def scatter(self, indices: Self, values: Self) -> Self:
        return self._with_native(self.native.scatter(indices.native, values.native))

    def shift(self, n: int, *, fill_value: NonNestedLiteral = None) -> Self:
        return self._with_native(self.native.shift(n, fill_value=fill_value))

    def slice(self, offset: int, length: int | None = None) -> Self:
        return self._with_native(self.native.slice(offset, length))

    # NOTE: Needs compat
    # https://github.com/narwhals-dev/narwhals/blob/c207fc096263ce174470240748e0c568f38f93e2/narwhals/_polars/series.py#L495-L505
    sort = todo()

    def sum(self) -> float:
        return self.native.sum()

    def unique(self, *, maintain_order: bool = False) -> Self:
        return self._with_native(self.native.unique(maintain_order=maintain_order))

    def zip_with(self, mask: Self, other: Self) -> Self:
        return self._with_native(self.native.zip_with(mask.native, other.native))

    @property
    def struct(self) -> SeriesStructNamespace:
        return SeriesStructNamespace(self)


class SeriesStructNamespace(StructNamespace["DataFrame", PolarsSeries]):
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
        return Namespace(self.version)._dataframe.from_native(df, self.version)

    @property
    def schema(self) -> Schema:
        return into_version(self.version).schema.from_polars(self.native.struct.schema)


PolarsSeries()
