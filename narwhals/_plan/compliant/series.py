from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan.compliant.typing import HasVersion
from narwhals._plan.typing import NativeSeriesT
from narwhals._utils import Version

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.series import Series
    from narwhals.dtypes import DType
    from narwhals.typing import Into1DArray, IntoDType, _1DArray

Incomplete: TypeAlias = Any


class CompliantSeries(HasVersion, Protocol[NativeSeriesT]):
    _native: NativeSeriesT
    _name: str

    def __len__(self) -> int:
        return len(self.native)

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
    def to_frame(self) -> Incomplete: ...
    def to_list(self) -> list[Any]: ...
    def to_narwhals(self) -> Series[NativeSeriesT]:
        from narwhals._plan.series import Series

        return Series[NativeSeriesT](self)

    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _1DArray: ...
