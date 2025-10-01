from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic

from narwhals._plan.typing import NativeSeriesT, NativeSeriesT_co
from narwhals._utils import Version
from narwhals.dependencies import is_pyarrow_chunked_array

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals.dtypes import DType


class Series(Generic[NativeSeriesT_co]):
    _compliant: CompliantSeries[NativeSeriesT_co]
    _version: ClassVar[Version] = Version.MAIN

    @property
    def version(self) -> Version:
        return self._version

    @property
    def dtype(self) -> DType:
        return self._compliant.dtype

    @property
    def name(self) -> str:
        return self._compliant.name

    def __init__(self, compliant: CompliantSeries[NativeSeriesT_co], /) -> None:
        self._compliant = compliant

    @classmethod
    def from_native(
        cls: type[Series[Any]], native: NativeSeriesT, name: str = "", /
    ) -> Series[NativeSeriesT]:
        if is_pyarrow_chunked_array(native):
            from narwhals._plan.arrow.series import ArrowSeries

            return cls(ArrowSeries.from_native(native, name, version=cls._version))

        raise NotImplementedError(type(native))

    def to_native(self) -> NativeSeriesT_co:
        return self._compliant.native

    def to_list(self) -> list[Any]:
        return self._compliant.to_list()

    def __iter__(self) -> Iterator[Any]:
        yield from self.to_native()


class SeriesV1(Series[NativeSeriesT_co]):
    _version: ClassVar[Version] = Version.V1
