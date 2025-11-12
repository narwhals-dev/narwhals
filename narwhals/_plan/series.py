from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic

from narwhals._plan.typing import NativeSeriesT, NativeSeriesT_co
from narwhals._utils import Implementation, Version, is_eager_allowed
from narwhals.dependencies import is_pyarrow_chunked_array

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import polars as pl
    from typing_extensions import Self

    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._typing import EagerAllowed, IntoBackend
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType, SizedMultiIndexSelector


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
    def from_iterable(
        cls: type[Series[Any]],
        values: Iterable[Any],
        *,
        name: str = "",
        dtype: IntoDType | None = None,
        backend: IntoBackend[EagerAllowed],
    ) -> Series[Any]:
        implementation = Implementation.from_backend(backend)
        if is_eager_allowed(implementation):
            if implementation is Implementation.PYARROW:
                from narwhals._plan import arrow as _arrow

                return cls(
                    _arrow.Series.from_iterable(
                        values, name=name, version=cls._version, dtype=dtype
                    )
                )
            raise NotImplementedError(implementation)
        else:  # pragma: no cover  # noqa: RET506
            msg = f"{implementation} support in Narwhals is lazy-only"
            raise ValueError(msg)

    @classmethod
    def from_native(
        cls: type[Series[Any]], native: NativeSeriesT, name: str = "", /
    ) -> Series[NativeSeriesT]:
        if is_pyarrow_chunked_array(native):
            from narwhals._plan import arrow as _arrow

            return cls(_arrow.Series.from_native(native, name, version=cls._version))

        raise NotImplementedError(type(native))

    def to_native(self) -> NativeSeriesT_co:
        return self._compliant.native

    def to_list(self) -> list[Any]:
        return self._compliant.to_list()

    def to_polars(self) -> pl.Series:
        return self._compliant.to_polars()

    def __iter__(self) -> Iterator[Any]:  # pragma: no cover
        yield from self.to_native()

    def alias(self, name: str) -> Self:
        return type(self)(self._compliant.alias(name))

    def __len__(self) -> int:
        return len(self._compliant)

    def gather(self, indices: SizedMultiIndexSelector[Self]) -> Self:  # pragma: no cover
        if len(indices) == 0:
            return self.slice(0, 0)
        rows = indices._compliant if isinstance(indices, Series) else indices
        return type(self)(self._compliant.gather(rows))

    def slice(self, offset: int, length: int | None = None) -> Self:  # pragma: no cover
        return type(self)(self._compliant.slice(offset=offset, length=length))

    def sort(
        self, *, descending: bool = False, nulls_last: bool = False
    ) -> Self:  # pragma: no cover
        result = self._compliant.sort(descending=descending, nulls_last=nulls_last)
        return type(self)(result)


class SeriesV1(Series[NativeSeriesT_co]):
    _version: ClassVar[Version] = Version.V1
