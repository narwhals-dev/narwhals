from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from narwhals._plan.compliant.typing import HasVersion
from narwhals._plan.typing import NativeSeriesT
from narwhals._utils import Version, _StoresNative

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl
    from typing_extensions import Self, TypeAlias

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

    def alias(self, name: str) -> Self:
        return self.from_native(self.native, name, version=self.version)

    def cast(self, dtype: IntoDType) -> Self: ...
    def cum_sum(self, *, reverse: bool = False) -> Self: ...
    def cum_count(self, *, reverse: bool = False) -> Self: ...
    def fill_null(self, value: NonNestedLiteral | Self) -> Self: ...
    def fill_null_with_strategy(
        self, strategy: FillNullStrategy, limit: int | None = None
    ) -> Self: ...
    def shift(self, n: int, *, fill_value: NonNestedLiteral = None) -> Self: ...
    def gather(
        self,
        indices: SizedMultiIndexSelector[NativeSeriesT] | _StoresNative[NativeSeriesT],
    ) -> Self: ...
    def has_nulls(self) -> bool: ...
    def is_empty(self) -> bool:
        return len(self) == 0

    def is_in(self, other: Self) -> Self: ...
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
