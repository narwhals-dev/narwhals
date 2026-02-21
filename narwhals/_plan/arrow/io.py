from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from io import BytesIO

    from typing_extensions import Unpack

    from narwhals._plan.arrow.typing import CSVReaderOptions, IOSource
    from narwhals.typing import FileSource


def read_parquet_schema(source: IOSource, /) -> pa.Schema:
    """Get the schema of a Parquet file without reading data.

    This has a direct path to Cython, and a single call before C++.
    """
    import pyarrow.parquet as pq

    reader = pq.ParquetReader()
    try:
        reader.open(source)
        schema = reader.schema_arrow
    finally:
        reader.close()
    return schema


def read_csv(source: FileSource, /, **kwds: Unpack[CSVReaderOptions]) -> pa.Table:
    import pyarrow.csv

    return pyarrow.csv.read_csv(super_normalize_path(source), **kwds)


def read_parquet(source: IOSource, /, **kwds: Any) -> pa.Table:
    import pyarrow.parquet as pq

    source = _normalize_io_source(source)
    if not kwds:
        with pq.ParquetFile(source) as f:
            return f.read()
    return pq.read_table(source, **kwds)


def _normalize_io_source(source: IOSource, /) -> str | BytesIO | pa.NativeFile:
    import io

    if not isinstance(source, (io.FileIO, io.BytesIO, pa.NativeFile)):
        return super_normalize_path(source)
    return source


def super_normalize_path(source: FileSource, /) -> str:
    r"""An extension of `nw._utils.normalize_path`, learning from `pyarrow.fs`, `fsspec`.

    Handles many more cases and the result is cached.

    - Preserves URI schemes
    - Unwraps `__fspath__`
    - Expands `~`
    - Resolves symlinks
    - Relative segments -> absolute
    - (Windows) `\\` -> `/`
    """
    if not isinstance(source, (str, Path)):
        source = os.fspath(source)
    return _normalize_path(source)


_URI_SCHEMES = (
    r"file://",
    r"mock://",
    r"s3fs://",
    r"gs://",
    r"gcs://",
    r"hdfs://",
    r"viewfs://",
    r"fsspec+",
    r"hf://",
)
"""These are not handled gracefully via `pathlib.Path`."""


def _has_uri_scheme(source: str, /) -> bool:
    return source.startswith(_URI_SCHEMES)


@lru_cache(maxsize=64)
def _normalize_path(source: str | Path, /) -> str:
    # NOTE: `__fspath__` is excluded from caching, as `os.PathLike`
    # doesn't require a hash or prevent mutation
    if isinstance(source, str) and _has_uri_scheme(source):
        return source
    return Path(source).expanduser().resolve().absolute().as_posix()
