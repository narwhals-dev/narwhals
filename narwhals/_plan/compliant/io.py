"""Minimal protocols for importing/exporting `*Frame`s from/to files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, overload

from narwhals._plan.compliant.typing import DataFrameT_co, LazyFrameT_co

if TYPE_CHECKING:
    from io import BytesIO


class ScanCsv(Protocol[LazyFrameT_co]):
    def scan_csv(self, source: str, /, **kwds: Any) -> LazyFrameT_co: ...


class ScanParquet(Protocol[LazyFrameT_co]):
    def scan_parquet(self, source: str, /, **kwds: Any) -> LazyFrameT_co: ...


class ReadCsv(Protocol[DataFrameT_co]):
    def read_csv(self, source: str, /, **kwds: Any) -> DataFrameT_co: ...


class ReadParquet(Protocol[DataFrameT_co]):
    def read_parquet(self, source: str, /, **kwds: Any) -> DataFrameT_co: ...


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


class LazyInput(
    ScanCsv[LazyFrameT_co], ScanParquet[LazyFrameT_co], Protocol[LazyFrameT_co]
):
    """Supports all `scan_*` methods, for lazily reading from files."""


class EagerInput(
    ReadCsv[DataFrameT_co], ReadParquet[DataFrameT_co], Protocol[DataFrameT_co]
):
    """Supports all `read_*` methods, for eagerly reading from files."""


class LazyOutput(SinkParquet, Protocol):
    """Supports all `sink_*` methods, for lazily writing to files."""


class EagerOutput(WriteCsv, WriteParquet, Protocol):
    """Supports all `write_*` methods, for eagerly writing to files."""
