from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from narwhals._plan.compliant.typing import HasVersion
from narwhals._plan.typing import NativeSeriesT
from narwhals._utils import Version, _StoresNative, unstable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import polars as pl
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.compliant.accessors import SeriesStructNamespace
    from narwhals._plan.compliant.dataframe import CompliantDataFrame
    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.series import Series
    from narwhals._typing import _EagerAllowedImpl
    from narwhals.dtypes import DType
    from narwhals.typing import (
        FillNullStrategy,
        Into1DArray,
        IntoDType,
        NonNestedLiteral,
        NumericLiteral,
        SizedMultiIndexSelector,
        TemporalLiteral,
        _1DArray,
    )

Incomplete: TypeAlias = Any


class CompliantSeries(HasVersion, Protocol[NativeSeriesT]):
    implementation: ClassVar[_EagerAllowedImpl]
    _native: NativeSeriesT
    _name: str

    def __len__(self) -> int:
        return len(self.native)

    def __add__(self, other: NumericLiteral | TemporalLiteral | Self, /) -> Self: ...
    def __and__(self, other: bool | Self, /) -> Self: ...
    def __eq__(self, other: NumericLiteral | TemporalLiteral | Self, /) -> Self: ...  # type: ignore[override]
    def __floordiv__(self, other: NumericLiteral | TemporalLiteral | Self, /) -> Self: ...
    def __ge__(self, other: NonNestedLiteral | Self, /) -> Self: ...
    def __gt__(self, other: NonNestedLiteral | Self, /) -> Self: ...
    def __invert__(self) -> Self: ...
    def __le__(self, other: NonNestedLiteral | Self, /) -> Self: ...
    def __lt__(self, other: NonNestedLiteral | Self, /) -> Self: ...
    def __mod__(self, other: NumericLiteral | TemporalLiteral | Self, /) -> Self: ...
    def __mul__(self, other: NumericLiteral | TemporalLiteral | Self, /) -> Self: ...
    def __ne__(self, other: NumericLiteral | TemporalLiteral | Self, /) -> Self: ...  # type: ignore[override]
    def __or__(self, other: bool | Self, /) -> Self: ...
    def __pow__(self, other: float | Self, /) -> Self: ...
    def __rfloordiv__(
        self, other: NumericLiteral | TemporalLiteral | Self, /
    ) -> Self: ...
    def __radd__(self, other: NumericLiteral | TemporalLiteral | Self, /) -> Self: ...
    def __rand__(self, other: bool | Self, /) -> Self: ...
    def __rmod__(self, other: NumericLiteral | TemporalLiteral | Self, /) -> Self: ...
    def __rmul__(self, other: NumericLiteral | TemporalLiteral | Self, /) -> Self: ...
    def __ror__(self, other: bool | Self, /) -> Self: ...
    def __rpow__(self, other: float | Self, /) -> Self: ...
    def __rsub__(self, other: NumericLiteral | TemporalLiteral | Self, /) -> Self: ...
    def __rtruediv__(self, other: NumericLiteral | TemporalLiteral | Self, /) -> Self: ...
    def __rxor__(self, other: bool | Self, /) -> Self: ...
    def __sub__(self, other: NumericLiteral | TemporalLiteral | Self, /) -> Self: ...
    def __truediv__(self, other: NumericLiteral | TemporalLiteral | Self, /) -> Self: ...
    def __xor__(self, other: bool | Self, /) -> Self: ...

    def len(self) -> int:
        return len(self.native)

    def not_(self) -> Self:
        return self.__invert__()

    def pow(self, exponent: float | Self) -> Self:
        return self.__pow__(exponent)

    def __narwhals_namespace__(self) -> Incomplete: ...
    def __narwhals_series__(self) -> Self:
        return self

    def _with_native(self, native: NativeSeriesT) -> Self:
        return self.from_native(native, self.name, version=self.version)

    @classmethod
    def from_iterable(
        cls,
        data: Iterable[Any],
        *,
        version: Version,
        name: str = "",
        dtype: IntoDType | None = None,
    ) -> Self: ...
    @classmethod
    def from_native(
        cls, native: NativeSeriesT, name: str = "", /, *, version: Version = Version.MAIN
    ) -> Self:
        obj = cls.__new__(cls)
        obj._native = native
        obj._name = name
        obj._version = version
        return obj

    @classmethod
    def from_numpy(
        cls, data: Into1DArray, name: str = "", /, *, version: Version = Version.MAIN
    ) -> Self: ...
    @property
    def dtype(self) -> DType: ...
    @property
    def name(self) -> str:
        return self._name

    @property
    def native(self) -> NativeSeriesT:
        return self._native

    def all(self) -> bool: ...
    def any(self) -> bool: ...
    def sum(self) -> float: ...
    def count(self) -> int: ...
    def alias(self, name: str) -> Self:
        return self.from_native(self.native, name, version=self.version)

    def cast(self, dtype: IntoDType) -> Self: ...
    def cum_count(self, *, reverse: bool = False) -> Self: ...
    def cum_max(self, *, reverse: bool = False) -> Self: ...
    def cum_min(self, *, reverse: bool = False) -> Self: ...
    def cum_prod(self, *, reverse: bool = False) -> Self: ...
    def cum_sum(self, *, reverse: bool = False) -> Self: ...
    def diff(self, n: int = 1) -> Self: ...
    def drop_nulls(self) -> Self: ...
    def drop_nans(self) -> Self: ...
    def explode(self, *, empty_as_null: bool = True, keep_nulls: bool = True) -> Self: ...
    def fill_nan(self, value: float | Self | None) -> Self: ...
    def fill_null(self, value: NonNestedLiteral | Self) -> Self: ...
    def fill_null_with_strategy(
        self, strategy: FillNullStrategy, limit: int | None = None
    ) -> Self: ...
    def shift(self, n: int, *, fill_value: NonNestedLiteral = None) -> Self: ...
    def gather(
        self,
        indices: SizedMultiIndexSelector[NativeSeriesT] | _StoresNative[NativeSeriesT],
    ) -> Self: ...
    def gather_every(self, n: int, offset: int = 0) -> Self: ...
    def has_nulls(self) -> bool: ...
    def null_count(self) -> int: ...
    def is_empty(self) -> bool:
        return len(self) == 0

    def is_in(self, other: Self) -> Self: ...
    def is_nan(self) -> Self: ...
    def is_null(self) -> Self: ...
    def is_not_nan(self) -> Self:
        return self.is_nan().__invert__()

    def is_not_null(self) -> Self:
        return self.is_null().__invert__()

    def rolling_mean(
        self, window_size: int, *, min_samples: int, center: bool = False
    ) -> Self: ...
    def rolling_std(
        self, window_size: int, *, min_samples: int, center: bool = False, ddof: int = 1
    ) -> Self: ...
    def rolling_sum(
        self, window_size: int, *, min_samples: int, center: bool = False
    ) -> Self: ...
    def rolling_var(
        self, window_size: int, *, min_samples: int, center: bool = False, ddof: int = 1
    ) -> Self: ...
    def sample_frac(
        self, fraction: float, *, with_replacement: bool = False, seed: int | None = None
    ) -> Self:
        n = int(len(self) * fraction)
        return self.sample_n(n, with_replacement=with_replacement, seed=seed)

    def sample_n(
        self, n: int = 1, *, with_replacement: bool = False, seed: int | None = None
    ) -> Self: ...
    def scatter(self, indices: Self, values: Self) -> Self: ...
    def slice(self, offset: int, length: int | None = None) -> Self: ...
    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self: ...
    def to_frame(self) -> Incomplete: ...
    def to_list(self) -> list[Any]: ...
    def to_narwhals(self) -> Series[NativeSeriesT]:
        from narwhals._plan.series import Series

        return Series[NativeSeriesT](self)

    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _1DArray: ...
    def to_polars(self) -> pl.Series: ...
    def unique(self, *, maintain_order: bool = False) -> Self: ...
    def zip_with(self, mask: Self, other: Self) -> Self: ...
    @unstable
    def hist(
        self,
        bins: Sequence[float] | None = None,
        *,
        bin_count: int | None = None,
        include_breakpoint: bool = True,
        include_category: bool = False,
        _compatibility_behavior: Literal["narwhals", "polars"] = "narwhals",
    ) -> CompliantDataFrame[Self, Incomplete, NativeSeriesT]:
        from narwhals._plan.expressions import col as ir_col

        expr = (
            ir_col(self.name)
            .to_narwhals(self.version)
            .hist(
                bins,
                bin_count=bin_count,
                include_breakpoint=include_breakpoint,
                include_category=include_category,
            )
        )
        df: DataFrame[Incomplete, NativeSeriesT] = (
            self.to_narwhals().to_frame().select(expr)
        )
        if not include_breakpoint and not include_category:
            if _compatibility_behavior == "narwhals":
                df = df.rename({self.name: "count"})
        else:
            df = df.to_series().struct.unnest()
        return df._compliant

    @property
    def struct(self) -> SeriesStructNamespace[Incomplete, Self]: ...
