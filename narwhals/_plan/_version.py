"""Wrapper around `narwhals._utils.Version`."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from narwhals._utils import Version as _NwVersion

if TYPE_CHECKING:
    from narwhals._plan.dataframe import DataFrame as NwDataFrame
    from narwhals._plan.expr import Expr as NwExpr
    from narwhals._plan.lazyframe import LazyFrame as NwLazyFrame
    from narwhals._plan.selectors import Selector as NwSelector
    from narwhals._plan.series import Series as NwSeries
    from narwhals.schema import Schema as NwSchema


__all__ = ("into_version",)


def into_version(version: _NwVersion | _HasVersion | _HasVersionClass, /) -> _Version:
    """Create a namespace for accessing versioned narwhals-level types.

    Rewraps the `Version` enum to use `narwhals._plan` package *and* cache the imports.

    Arguments:
        version: A `Version` or an object with `version: Version`.
    """
    return _Version(version if isinstance(version, _NwVersion) else version.version)


class _Version:
    __slots__ = ("_version",)

    def __init__(self, version: _NwVersion, /) -> None:
        self._version: _NwVersion = version

    @property
    def dataframe(self) -> type[NwDataFrame[Any, Any]]:
        return _import_dataframe(self._version)

    @property
    def lazyframe(self) -> type[NwLazyFrame[Any]]:
        return _import_lazyframe(self._version)

    @property
    def series(self) -> type[NwSeries[Any]]:  # pragma: no cover
        return _import_series(self._version)

    @property
    def schema(self) -> type[NwSchema]:
        """TODO @dangotbanned: Upstream this, it has the same meaning as `Version.dtypes`."""
        return _import_schema(self._version)

    @property
    def selector(self) -> type[NwSelector]:
        return _import_selector(self._version)

    @property
    def expr(self) -> type[NwExpr]:
        return _import_expr(self._version)


# NOTE: Another choice here is to use string import paths, inside something like:
#    `Mapping[Version, Mapping[Literal["DataFrame", "LazyFrame", ...],  Literal["narwhals._plan.dataframe.DataFrame", ...]]]`
# Although it could save some lines, it would require manually syncing any time these symbols change location.
# It is expected at-least two major changes will eventually come:
#    1. The version'd equivalent(s) of each type are developed enough to add `.stable.v<n>` packages.
#    2. The entire `_plan` package is promoted to the top-level.
# This solution disables formatting to get a similar density of code
# - but has the benefit of refactoring code actions syncing the paths for us.
# fmt: off
@cache
def _import_dataframe(version: _NwVersion, /) -> type[NwDataFrame[Any, Any]]:
    if version is _NwVersion.MAIN:
        from narwhals._plan.dataframe import DataFrame as NwDataFrame
        return NwDataFrame
    raise _not_implemented(version)  # pragma: no cover
@cache
def _import_lazyframe(version: _NwVersion, /) -> type[NwLazyFrame[Any]]:
    if version is _NwVersion.MAIN:
        from narwhals._plan.lazyframe import LazyFrame as NwLazyFrame
        return NwLazyFrame
    raise _not_implemented(version)  # pragma: no cover
@cache
def _import_series(version: _NwVersion, /) -> type[NwSeries[Any]]:  # pragma: no cover
    if version is _NwVersion.MAIN:
        from narwhals._plan.series import Series as NwSeries
        return NwSeries
    if version is _NwVersion.V1:
        from narwhals._plan.series import SeriesV1 as NwSeriesV1
        return NwSeriesV1
    raise _not_implemented(version)
@cache
def _import_schema(version: _NwVersion, /) -> type[NwSchema]:
    if version is _NwVersion.MAIN:
        from narwhals.schema import Schema as NwSchema
        return NwSchema
    if version is _NwVersion.V1:  # pragma: no cover
        from narwhals.stable.v1 import Schema as NwSchemaV1
        return NwSchemaV1
    from narwhals.stable.v2 import Schema as NwSchemaV2  # pragma: no cover
    return NwSchemaV2  # pragma: no cover
@cache
def _import_selector(version: _NwVersion, /) -> type[NwSelector]:
    from narwhals._plan.selectors import Selector, SelectorV1
    if version is _NwVersion.MAIN:
        return Selector
    if version is _NwVersion.V1:  # pragma: no cover
        return SelectorV1
    raise _not_implemented(version)  # pragma: no cover
@cache
def _import_expr(version: _NwVersion, /) -> type[NwExpr]:
    from narwhals._plan.expr import Expr, ExprV1
    if version is _NwVersion.MAIN:
        return Expr
    if version is _NwVersion.V1:  # pragma: no cover
        return ExprV1
    raise _not_implemented(version)  # pragma: no cover
# fmt: on


class _HasVersion(Protocol):
    @property
    def version(self) -> _NwVersion: ...


class _HasVersionClass(Protocol):
    version: ClassVar[_NwVersion]


def _not_implemented(version: _NwVersion) -> NotImplementedError:  # pragma: no cover
    msg = f"{version!r} is not yet implemented for `narwhals._plan`"
    return NotImplementedError(msg)
