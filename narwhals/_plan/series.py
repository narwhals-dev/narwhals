from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic

from narwhals._plan.typing import NativeSeriesT
from narwhals._utils import Version
from narwhals.dependencies import is_pyarrow_chunked_array

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals.dtypes import DType
    from narwhals.typing import NativeSeries


class Series(Generic[NativeSeriesT]):
    _compliant: CompliantSeries[NativeSeriesT]
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

    # NOTE: Gave up on trying to get typing working for now
    @classmethod
    def from_native(
        cls, native: NativeSeries, name: str = "", /
    ) -> Series[pa.ChunkedArray[Any]]:
        if is_pyarrow_chunked_array(native):
            from narwhals._plan.arrow.series import ArrowSeries

            return ArrowSeries.from_native(
                native, name, version=cls._version
            ).to_narwhals()

        raise NotImplementedError(type(native))

    @classmethod
    def _from_compliant(cls, compliant: CompliantSeries[NativeSeriesT], /) -> Self:
        obj = cls.__new__(cls)
        obj._compliant = compliant
        return obj

    def to_native(self) -> NativeSeriesT:
        return self._compliant.native

    def to_list(self) -> list[Any]:
        return self._compliant.to_list()

    def __iter__(self) -> Iterator[Any]:
        yield from self.to_native()


class SeriesV1(Series[NativeSeriesT]):
    _version: ClassVar[Version] = Version.V1
