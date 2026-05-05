"""Minimal protocols for importing/exporting `*Frame`s from/to files and guards to test for them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan.compliant.typing import Native_co as LF
from narwhals._utils import _hasattr_static

if TYPE_CHECKING:
    from io import BytesIO

    from typing_extensions import TypeIs

    from narwhals._plan.compliant.lazyframe import CompliantLazyFrame as LazyFrame
    from narwhals.schema import Schema

__all__ = (
    "LazyInput",
    "LazyOutput",
    "ScanCsv",
    "ScanParquet",
    "SinkParquet",
    "can_read_csv_schema",
    "can_read_parquet_schema",
    "can_scan_csv",
    "can_scan_parquet",
)


# fmt: off
class ScanCsv(Protocol[LF]):
    def scan_csv(self, source: str, /, **kwds: Any) -> LazyFrame[LF]: ...
    def read_csv_schema(self, source: str, /, **kwds: Any) -> Schema: ...
class ScanParquet(Protocol[LF]):
    def scan_parquet(self, source: str, /, **kwds: Any) -> LazyFrame[LF]: ...
    def read_parquet_schema(self, source: str, /) -> Schema: ...
class SinkParquet(Protocol):
    def sink_parquet(self, target: str | BytesIO, /) -> None: ...
class LazyInput(ScanCsv[LF], ScanParquet[LF], Protocol[LF]):
    """Supports all `scan_*` methods, for lazily reading from files."""
class LazyOutput(SinkParquet, Protocol):
    """Supports all `sink_*` methods, for lazily writing to files."""

def can_scan_csv(obj: ScanCsv[LF] | Any) -> TypeIs[ScanCsv[LF]]:
    return _hasattr_static(obj, "scan_csv")
def can_scan_parquet(obj: ScanParquet[LF] | Any) -> TypeIs[ScanParquet[LF]]:
    return _hasattr_static(obj, "scan_parquet")
def can_read_csv_schema(obj: Any) -> TypeIs[ScanCsv[Any]]:
    return _hasattr_static(obj, "read_csv_schema")
def can_read_parquet_schema(obj: Any) -> TypeIs[ScanParquet[Any]]:
    return _hasattr_static(obj, "read_parquet_schema")
# fmt: on
