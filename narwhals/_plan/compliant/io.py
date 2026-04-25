"""Minimal protocols for importing/exporting `*Frame`s from/to files and guards to test for them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, overload

from narwhals._plan.compliant.typing import (
    Native_co as LF,
    NativeDataFrameT_co as DF,
    NativeSeriesT_co as S,
)
from narwhals._utils import _hasattr_static

if TYPE_CHECKING:
    from io import BytesIO

    from typing_extensions import TypeIs

    from narwhals._plan.compliant.dataframe import CompliantDataFrame as DataFrame
    from narwhals._plan.compliant.lazyframe import CompliantLazyFrame as LazyFrame
    from narwhals.schema import Schema

__all__ = [
    "EagerInput",
    "EagerOutput",
    "LazyInput",
    "LazyOutput",
    "ReadCsv",
    "ReadParquet",
    "ScanCsv",
    "ScanParquet",
    "SinkParquet",
    "WriteCsv",
    "WriteParquet",
    "can_read_csv",
    "can_read_csv_schema",
    "can_read_parquet",
    "can_read_parquet_schema",
    "can_scan_csv",
    "can_scan_parquet",
    "can_sink_parquet",
    "can_write_csv",
    "can_write_parquet",
]


# fmt: off
class ScanCsv(Protocol[LF]):
    def scan_csv(self, source: str, /, **kwds: Any) -> LazyFrame[LF]: ...
    def read_csv_schema(self, source: str, /, **kwds: Any) -> Schema: ...
class ScanParquet(Protocol[LF]):
    def scan_parquet(self, source: str, /, **kwds: Any) -> LazyFrame[LF]: ...
    def read_parquet_schema(self, source: str, /) -> Schema: ...
class ReadCsv(Protocol[DF, S]):
    def read_csv(self, source: str, /, **kwds: Any) -> DataFrame[DF, S]: ...
class ReadParquet(Protocol[DF, S]):
    def read_parquet(self, source: str, /, **kwds: Any) -> DataFrame[DF, S]: ...
class SinkParquet(Protocol):
    def sink_parquet(self, target: str | BytesIO, /) -> None: ...
class WriteCsv(Protocol):
    @overload
    def write_csv(self, target: None, /) -> str: ...
    @overload
    def write_csv(self, target: str | BytesIO, /) -> None: ...
    def write_csv(self, target: str | BytesIO | None, /) -> str | None: ...
class WriteParquet(Protocol):
    def write_parquet(self, target: str | BytesIO, /) -> None: ...
class LazyInput(ScanCsv[LF], ScanParquet[LF], Protocol[LF]):
    """Supports all `scan_*` methods, for lazily reading from files."""
class EagerInput(ReadCsv[DF, S], ReadParquet[DF, S], Protocol[DF, S]):
    """Supports all `read_*` methods, for eagerly reading from files."""
class LazyOutput(SinkParquet, Protocol):
    """Supports all `sink_*` methods, for lazily writing to files."""
class EagerOutput(WriteCsv, WriteParquet, Protocol):
    """Supports all `write_*` methods, for eagerly writing to files."""


def can_read_csv(obj: ReadCsv[DF, S] | Any) -> TypeIs[ReadCsv[DF, S]]:
    return _hasattr_static(obj, "read_csv")
def can_read_parquet(obj: ReadParquet[DF, S] | Any) -> TypeIs[ReadParquet[DF, S]]:
    return _hasattr_static(obj, "read_parquet")
def can_scan_csv(obj: ScanCsv[LF] | Any) -> TypeIs[ScanCsv[LF]]:
    return _hasattr_static(obj, "scan_csv")
def can_scan_parquet(obj: ScanParquet[LF] | Any) -> TypeIs[ScanParquet[LF]]:
    return _hasattr_static(obj, "scan_parquet")
def can_read_csv_schema(obj: Any) -> TypeIs[ScanCsv[Any]]:
    return _hasattr_static(obj, "read_csv_schema")
def can_read_parquet_schema(obj: Any) -> TypeIs[ScanParquet[Any]]:
    return _hasattr_static(obj, "read_parquet_schema")
def can_write_csv(obj: Any) -> TypeIs[WriteCsv]:
    return _hasattr_static(obj, "write_csv")
def can_write_parquet(obj: Any) -> TypeIs[WriteParquet]:
    return _hasattr_static(obj, "write_parquet")
def can_sink_parquet(obj: Any) -> TypeIs[SinkParquet]:
    return _hasattr_static(obj, "sink_parquet")
# fmt: on
