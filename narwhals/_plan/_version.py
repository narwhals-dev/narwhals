from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._utils import Version as _NwVersion

if TYPE_CHECKING:
    from narwhals._plan.dataframe import DataFrame as NwDataFrame
    from narwhals._plan.lazyframe import LazyFrame as NwLazyFrame
    from narwhals._plan.series import Series as NwSeries

__all__ = ["into_version"]


def into_version(version: _NwVersion, /) -> _Version:
    """Rewraps `nw._utils.Version` to use `nw._plan` namespace."""
    return _Version(version)


class _Version:
    __slots__ = ("_version",)

    def __init__(self, version: _NwVersion, /) -> None:
        self._version: _NwVersion = version

    def _not_implemented(self) -> NotImplementedError:
        msg = f"{self._version!r} is not yet implemented for `narwhals._plan`"
        return NotImplementedError(msg)

    @property
    def dataframe(self) -> type[NwDataFrame[Any, Any]]:
        if self._version is _NwVersion.MAIN:
            from narwhals._plan.dataframe import DataFrame as NwDataFrame

            return NwDataFrame
        raise self._not_implemented()

    @property
    def lazyframe(self) -> type[NwLazyFrame[Any]]:
        if self._version is _NwVersion.MAIN:
            from narwhals._plan.lazyframe import LazyFrame as NwLazyFrame

            return NwLazyFrame
        raise self._not_implemented()

    @property
    def series(self) -> type[NwSeries[Any]]:
        if self._version is _NwVersion.MAIN:
            from narwhals._plan.series import Series as NwSeries

            return NwSeries
        if self._version is _NwVersion.V1:
            from narwhals._plan.series import SeriesV1 as NwSeriesV1

            return NwSeriesV1
        raise self._not_implemented()
