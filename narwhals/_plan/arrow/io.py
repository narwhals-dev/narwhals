from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa  # ignore-banned-import

from narwhals._utils import unstable

if TYPE_CHECKING:
    from io import BytesIO

    from pyarrow import fs
    from typing_extensions import TypeAlias, Unpack

    from narwhals._plan.arrow.typing import CSVReaderOptions, IOSource
    from narwhals.typing import FileSource


FileSystemPath: TypeAlias = "tuple[fs.FileSystem, str]"
"""Return type of [`FileSystem.from_uri`].

Where `path` may have been transformed from the original `source: FileSource`:

    (filesystem, path)

[`FileSystem.from_uri`]: https://arrow.apache.org/docs/python/generated/pyarrow.fs.FileSystem.html#pyarrow.fs.FileSystem.from_uri
"""


def read_csv_schema(source: FileSource, /, **kwds: Unpack[CSVReaderOptions]) -> pa.Schema:
    """Infer the schema of a Csv file, by default using the first block.

    Extends the default [`open_csv`] with a subset of the *file source inference* from [`dataset`].

    See [`CSVStreamingReader`] for configuring the *type inference* window.

    [`open_csv`]: https://github.com/apache/arrow/blob/0cf32b23c361d174742befc201617b56040fc095/python/pyarrow/_csv.pyx#L1274-L1318
    [`dataset`]: https://github.com/apache/arrow/blob/0cf32b23c361d174742befc201617b56040fc095/python/pyarrow/dataset.py#L580-L813
    [`CSVStreamingReader`]: https://arrow.apache.org/docs/cpp/api/formats.html#_CPPv4N5arrow3csv15StreamingReaderE
    """
    source = super_normalize_path(source)
    if not _has_uri_scheme(source):
        import pyarrow.csv as pcsv

        with pcsv.open_csv(source, **kwds) as reader:
            return reader.schema
    import pyarrow.dataset as ds

    filesystem, path = file_system_path(source)
    return ds.CsvFileFormat(**kwds).inspect(path, filesystem)


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


def file_system_path(source: FileSource, /) -> FileSystemPath:
    """Prepare a *potentially* remote file for IO.

    Lightweight adaptation of logic from [`pyarrow.dataset.dataset`].

    [`pyarrow.dataset.dataset`]: https://github.com/apache/arrow/blob/0cf32b23c361d174742befc201617b56040fc095/python/pyarrow/dataset.py#L580-L813
    """
    from pyarrow import fs

    return fs.FileSystem.from_uri(super_normalize_path(source))


@unstable
def file_info(source: FileSystemPath | FileSource, /) -> fs.FileInfo:
    # TODO @dangotbanned: Utilize file metadata for `scan_*` schema caching
    if not isinstance(source, tuple):
        source = file_system_path(source)
    return source[0].get_file_info(source[1])


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
