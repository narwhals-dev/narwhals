"""Minimal protocols for importing/exporting `*Frame`s from/to files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from io import BytesIO

    from typing_extensions import Self

    from narwhals.schema import Schema

__all__ = ("LazyIO", "LazyInput", "LazyOutput", "ReadSchema")


# fmt: off
class LazyInput(Protocol):
    """Supports all `scan_*` methods, for lazily reading from files."""
    __slots__ = ()
    @classmethod
    def scan_csv(cls, source: str, /, **kwds: Any) -> Self: ...
    @classmethod
    def scan_parquet(cls, source: str, /, **kwds: Any) -> Self: ...
class LazyOutput(Protocol):
    """Supports all `sink_*` methods, for lazily writing to files."""
    __slots__ = ()
    def sink_parquet(self, target: str | BytesIO, /) -> None: ...
class LazyIO(LazyInput, LazyOutput, Protocol):
    """Supports all `scan_*`, `sink_*` methods."""
    __slots__ = ()
class ReadSchema(Protocol):
    __slots__ = ()
    def read_csv_schema(self, source: str, /, **kwds: Any) -> Schema: ...
    def read_parquet_schema(self, source: str, /, **kwds: Any) -> Schema: ...
# fmt: on
