"""Minimal protocols for importing/exporting `*Frame`s from/to files and guards to test for them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._utils import _hasattr_static

if TYPE_CHECKING:
    from io import BytesIO

    from typing_extensions import Self, TypeIs

    from narwhals.schema import Schema

__all__ = (
    "LazyIO",
    "LazyInput",
    "LazyOutput",
    "can_read_csv_schema",
    "can_read_parquet_schema",
)


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
class ReadCsvSchema(Protocol):
    __slots__ = ()
    def read_csv_schema(self, source: str, /, **kwds: Any) -> Schema: ...
class ReadParquetSchema(Protocol):
    __slots__ = ()
    def read_parquet_schema(self, source: str, /) -> Schema: ...

def can_read_csv_schema(obj: Any) -> TypeIs[ReadCsvSchema]:
    return _hasattr_static(obj, "read_csv_schema")
def can_read_parquet_schema(obj: Any) -> TypeIs[ReadParquetSchema]:
    return _hasattr_static(obj, "read_parquet_schema")
# fmt: on
