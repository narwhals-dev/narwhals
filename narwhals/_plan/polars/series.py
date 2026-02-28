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
)
from narwhals._polars.utils import BACKEND_VERSION
from narwhals._utils import Implementation, Version
from narwhals.dependencies import is_numpy_array_1d

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.polars.dataframe import (
        PolarsDataFrame,
        PolarsDataFrame as DataFrame,
    )
    from narwhals.dtypes import DType
    from narwhals.schema import Schema
    from narwhals.typing import Into1DArray, IntoDType, _1DArray

Incomplete: TypeAlias = Any


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
    def from_native(
        cls, native: pl.Series, name: str = "", /, *, version: Version = Version.MAIN
    ) -> Self:
        obj = cls.__new__(cls)
        obj._native = native if not name else native.alias(name)
        obj._version = version
        return obj

    def _with_native(self, native: pl.Series) -> Self:
        return self.from_native(native, version=self.version)

    def to_list(self) -> list[Any]:
        return self.native.to_list()

    def to_polars(self) -> pl.Series:
        return self.native

    def cast(self, dtype: IntoDType) -> Self:
        result = self.native.cast(dtype_to_native(dtype, self._version))
        return self._with_native(result)

    @classmethod
    def from_numpy(
        cls, data: Into1DArray, name: str = "", /, *, version: Version = Version.MAIN
    ) -> Self:
        native = pl.Series(data if is_numpy_array_1d(data) else [data])
        return cls.from_native(native, name, version=version)

    def has_nulls(self) -> bool:
        return self.native.has_nulls()

    def is_in(self, other: Self) -> Self:
        return self._with_native(self.native.is_in(other.native))

    def is_nan(self) -> Self:
        return self._with_native(self.native.is_nan())

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
    __rfloordiv__ = todo()
    __rmod__ = todo()
    __rmul__ = todo()
    __ror__ = todo()
    __rpow__ = todo()
    __rsub__ = todo()
    __rtruediv__ = todo()
    __rxor__ = todo()
    __sub__ = todo()
    __truediv__ = todo()
    __xor__ = todo()
    all = todo()
    any = todo()
    count = todo()
    cum_count = todo()
    cum_max = todo()
    cum_min = todo()
    cum_prod = todo()
    cum_sum = todo()
    diff = todo()
    drop_nans = todo()
    drop_nulls = todo()
    explode = todo()
    fill_nan = todo()
    fill_null = todo()
    fill_null_with_strategy = todo()
    first = todo()
    from_iterable = todo()
    gather = todo()
    gather_every = todo()
    last = todo()
    null_count = todo()
    rolling_mean = todo()
    rolling_std = todo()
    rolling_sum = todo()
    rolling_var = todo()
    sample_n = todo()
    scatter = todo()
    shift = todo()
    slice = todo()
    sort = todo()
    sum = todo()
    unique = todo()
    zip_with = todo()

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
