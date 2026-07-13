from __future__ import annotations

import typing as _t

from narwhals._plan.compliant import plugins as _plugins

if _t.TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import TypeIs

    from narwhals._native import NativeDuckDB
    from narwhals._plan.duckdb.classes import DuckDBClasses

__all__ = ("DuckDBPlugin",)


@_t.final
class DuckDBPlugin(_plugins.Builtin["DuckDBClasses", _t.Any, "NativeDuckDB", _t.Any]):
    __slots__ = ()
    implementation = _plugins.Implementation.DUCKDB
    requirements = ("duckdb",)

    def is_native(self, obj: _t.Any) -> TypeIs[NativeDuckDB]:
        import duckdb

        return isinstance(obj, (duckdb.DuckDBPyRelation,))

    def native_lazyframe_classes(self) -> Iterator[type[NativeDuckDB]]:
        import duckdb

        yield duckdb.DuckDBPyRelation

    def native_series_classes(self) -> Iterator[type[_t.Any]]:
        yield from ()

    def native_dataframe_classes(self) -> Iterator[type[_t.Any]]:
        yield from ()

    @property
    def __narwhals_classes__(self) -> DuckDBClasses:
        from narwhals._plan.duckdb.classes import DuckDBClasses

        return DuckDBClasses()


plugin = DuckDBPlugin()
