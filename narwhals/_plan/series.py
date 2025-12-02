from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Generic

from narwhals._plan._guards import is_series
from narwhals._plan.typing import NativeSeriesT, NativeSeriesT_co, OneOrIterable
from narwhals._utils import (
    Implementation,
    Version,
    generate_repr,
    is_eager_allowed,
    qualified_type_name,
)
from narwhals.dependencies import is_pyarrow_chunked_array
from narwhals.exceptions import ShapeError

if TYPE_CHECKING:
    from collections.abc import Iterator

    import polars as pl
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.dataframe import DataFrame
    from narwhals._typing import EagerAllowed, IntoBackend, _EagerAllowedImpl
    from narwhals.dtypes import DType
    from narwhals.typing import (
        IntoDType,
        NonNestedLiteral,
        NumericLiteral,
        SizedMultiIndexSelector,
        TemporalLiteral,
    )

Incomplete: TypeAlias = Any


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

    @property
    def implementation(self) -> _EagerAllowedImpl:
        return self._compliant.implementation

    @property
    def shape(self) -> tuple[int]:
        return (self._compliant.len(),)

    def __init__(self, compliant: CompliantSeries[NativeSeriesT_co], /) -> None:
        self._compliant = compliant

    def __repr__(self) -> str:
        return generate_repr(f"nw.{type(self).__name__}", self.to_native().__repr__())

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

    # NOTE: `Incomplete` until `CompliantSeries` can avoid a cyclic dependency back to `CompliantDataFrame`
    # Currently an issue on `main` and leads to a lot of intermittent warnings
    def to_frame(self) -> DataFrame[Incomplete, NativeSeriesT_co]:
        import narwhals._plan.dataframe as _df

        # NOTE: Missing placeholder for `DataFrameV1`
        return _df.DataFrame(self._compliant.to_frame())

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

    def gather_every(self, n: int, offset: int = 0) -> Self:
        return type(self)(self._compliant.gather_every(n, offset))

    def has_nulls(self) -> bool:  # pragma: no cover
        return self._compliant.has_nulls()

    def slice(self, offset: int, length: int | None = None) -> Self:
        return type(self)(self._compliant.slice(offset=offset, length=length))

    def sort(
        self, *, descending: bool = False, nulls_last: bool = False
    ) -> Self:  # pragma: no cover
        result = self._compliant.sort(descending=descending, nulls_last=nulls_last)
        return type(self)(result)

    def is_empty(self) -> bool:  # pragma: no cover
        return self._compliant.is_empty()

    def _unwrap_compliant(
        self, other: Series[Any], /
    ) -> CompliantSeries[NativeSeriesT_co]:
        compliant = other._compliant
        if isinstance(compliant, type(self._compliant)):
            return compliant
        msg = f"Expected {qualified_type_name(self._compliant)!r}, got {qualified_type_name(compliant)!r}"
        raise NotImplementedError(msg)

    def _parse_into_compliant(
        self, other: Series[Any] | Iterable[Any], /
    ) -> CompliantSeries[NativeSeriesT_co]:
        if is_series(other):
            return self._unwrap_compliant(other)
        return self._compliant.from_iterable(other, version=self.version)

    def scatter(
        self,
        indices: Self | OneOrIterable[int],
        values: Self | OneOrIterable[NonNestedLiteral],
    ) -> Self:
        if not isinstance(indices, Iterable):
            indices = [indices]
        indices_ = self._parse_into_compliant(indices)
        if indices_.is_empty():
            return self
        if not is_series(values) and (
            not isinstance(values, Iterable) or isinstance(values, str)
        ):
            values = [values]
        result = self._compliant.scatter(indices_, self._parse_into_compliant(values))
        return type(self)(result)

    def is_in(self, other: Iterable[Any]) -> Self:
        return type(self)(self._compliant.is_in(self._parse_into_compliant(other)))

    def null_count(self) -> int:
        return self._compliant.null_count()

    def fill_nan(self, value: float | Self | None) -> Self:
        other = self._unwrap_compliant(value) if is_series(value) else value
        return type(self)(self._compliant.fill_nan(other))

    def sample(
        self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        with_replacement: bool = False,
        seed: int | None = None,
    ) -> Self:
        if n is not None and fraction is not None:
            msg = "cannot specify both `n` and `fraction`"
            raise ValueError(msg)
        s = self._compliant
        if fraction is not None:
            result = s.sample_frac(fraction, with_replacement=with_replacement, seed=seed)
        elif n is None:
            result = s.sample_n(with_replacement=with_replacement, seed=seed)
        elif not with_replacement and n > len(self):
            msg = "cannot take a larger sample than the total population when `with_replacement=false`"
            raise ShapeError(msg)
        else:
            result = s.sample_n(n, with_replacement=with_replacement, seed=seed)
        return type(self)(result)

    def __eq__(self, other: NumericLiteral | TemporalLiteral | Self) -> Self:  # type: ignore[override]
        other_ = self._unwrap_compliant(other) if is_series(other) else other
        return type(self)(self._compliant.__eq__(other_))

    def all(self) -> bool:
        return self._compliant.all()

    def any(self) -> bool:  # pragma: no cover
        return self._compliant.any()

    def unique(self, *, maintain_order: bool = False) -> Self:  # pragma: no cover
        return type(self)(self._compliant.unique(maintain_order=maintain_order))


class SeriesV1(Series[NativeSeriesT_co]):
    _version: ClassVar[Version] = Version.V1
