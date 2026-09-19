from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Any, Literal

import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    PANDAS_VERSION,
    Constructor,
    assert_equal_data,
    pyspark_session,
    sqlframe_session,
)

pytest.importorskip("polars")
pytest.importorskip("pyarrow")
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path
    from types import ModuleType
    from typing import TypeAlias

    from narwhals._typing import EagerAllowed, _LazyOnly, _SparkLike
    from narwhals.typing import FileSource

    Factory: TypeAlias = pytest.TempPathFactory

IOSourceKind: TypeAlias = Literal["str", "Path", "PathLike"]

data: Mapping[str, Any] = {"a": [1, 2, 3], "b": [4.5, 6.7, 8.9], "z": ["x", "y", "w"]}
skipif_pandas_lt_1_5 = pytest.mark.skipif(
    PANDAS_VERSION < (1, 5), reason="too old for pyarrow"
)
lazy_core_backend = pytest.mark.parametrize("backend", ["duckdb", "ibis", "sqlframe"])
spark_like_backend = pytest.mark.parametrize("backend", ["pyspark", "sqlframe"])
# Backends whose native reader cannot take a file-like object.
path_only_backend = pytest.mark.parametrize("backend", ["dask", "ibis", "sqlframe"])


class MockPathLike:
    def __init__(self, path: Path) -> None:
        self._super_secret: Path = path

    def __fspath__(self) -> str:
        return self._super_secret.__fspath__()


def _into_file_source(source: Path, which: IOSourceKind, /) -> FileSource:
    mapping: Mapping[IOSourceKind, FileSource] = {
        "str": str(source),
        "Path": source,
        "PathLike": MockPathLike(source),
    }
    return mapping[which]


def _path(factory: Factory, name: str, /) -> Path:
    # NOTE: Generates a path on windows that contains `\\n` and `\\t`
    # See https://github.com/narwhals-dev/narwhals/issues/3422
    tmp_dir = factory.mktemp("newline")
    sub_dir = tmp_dir / "tab"
    sub_dir.mkdir(exist_ok=True)
    return sub_dir / name


@pytest.fixture(scope="module", params=["str", "Path", "PathLike"])
def csv_path(tmp_path_factory: Factory, request: pytest.FixtureRequest) -> FileSource:
    fp = _path(tmp_path_factory, "file.csv")
    pl.DataFrame(data).write_csv(fp)
    return _into_file_source(fp, request.param)


@pytest.fixture(scope="module", params=["str", "Path", "PathLike"])
def csv_path_sep(tmp_path_factory: Factory, request: pytest.FixtureRequest) -> FileSource:
    fp = _path(tmp_path_factory, "file.csv")
    pl.DataFrame(data).write_csv(fp, separator="|")
    return _into_file_source(fp, request.param)


@pytest.fixture(scope="module", params=["str", "Path", "PathLike"])
def parquet_path(tmp_path_factory: Factory, request: pytest.FixtureRequest) -> FileSource:
    fp = _path(tmp_path_factory, "file.parquet")
    pl.DataFrame(data).write_parquet(fp)
    return _into_file_source(fp, request.param)


def assert_equal_eager(result: nw.DataFrame[Any]) -> None:
    assert_equal_data(result, data)
    assert isinstance(result, nw.DataFrame)


def assert_equal_lazy(result: nw.LazyFrame[Any]) -> None:
    assert_equal_data(result, data)
    assert isinstance(result, nw.LazyFrame)


def native_namespace(cb: Constructor, /) -> ModuleType:
    return nw.get_native_namespace(nw.from_native(cb(data)))  # type: ignore[no-any-return]


def _session_kwargs(name: str, /) -> dict[str, Any]:
    """Spark-like backends need a `session`, keyed off a backend name or constructor repr.

    Worth passing even when the call is rejected before the reader is reached: it keeps
    those tests failing on the behaviour under test, not on a missing session.
    """
    for backend, session in ("sqlframe", sqlframe_session), ("pyspark", pyspark_session):
        if backend in name:
            return {"session": session()}
    return {}


def test_read_csv(
    csv_path: FileSource, csv_path_sep: FileSource, eager_backend: EagerAllowed
) -> None:
    assert_equal_eager(nw.read_csv(csv_path, backend=eager_backend))
    assert_equal_eager(nw.read_csv(csv_path_sep, backend=eager_backend, separator="|"))


@skipif_pandas_lt_1_5
def test_read_csv_kwargs(csv_path: FileSource) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd
    from pyarrow import csv

    assert_equal_eager(nw.read_csv(csv_path, backend=pd, engine="pyarrow"))
    assert_equal_eager(
        nw.read_csv(
            csv_path, backend="pyarrow", parse_options=csv.ParseOptions(delimiter=",")
        )
    )


@lazy_core_backend
def test_read_csv_raise_with_lazy(backend: _LazyOnly) -> None:
    pytest.importorskip(backend)
    with pytest.raises(ValueError, match="Expected eager backend, found"):
        nw.read_csv("unused.csv", backend=backend)  # type: ignore[arg-type]


def test_scan_csv(
    csv_path: FileSource, csv_path_sep: FileSource, constructor: Constructor
) -> None:
    kwargs = _session_kwargs(str(constructor))
    if kwargs:
        kwargs.update(inferSchema=True, header=True)
    backend = native_namespace(constructor)
    assert_equal_lazy(nw.scan_csv(csv_path, backend=backend, **kwargs))
    assert_equal_lazy(nw.scan_csv(csv_path_sep, backend=backend, separator="|", **kwargs))


@skipif_pandas_lt_1_5
def test_scan_csv_kwargs(csv_path: FileSource) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    assert_equal_data(nw.scan_csv(csv_path, backend=pd, engine="pyarrow"), data)


@skipif_pandas_lt_1_5
def test_read_parquet(parquet_path: FileSource, eager_backend: EagerAllowed) -> None:
    assert_equal_eager(nw.read_parquet(parquet_path, backend=eager_backend))


@skipif_pandas_lt_1_5
def test_read_parquet_kwargs(parquet_path: FileSource) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    assert_equal_eager(nw.read_parquet(parquet_path, backend=pd, engine="pyarrow"))


@lazy_core_backend
def test_read_parquet_raise_with_lazy(backend: _LazyOnly) -> None:
    pytest.importorskip(backend)
    with pytest.raises(ValueError, match="Expected eager backend, found"):
        nw.read_parquet("unused.parquet", backend=backend)  # type: ignore[arg-type]


@skipif_pandas_lt_1_5
def test_scan_parquet(parquet_path: FileSource, constructor: Constructor) -> None:
    kwargs = _session_kwargs(str(constructor))
    if kwargs:
        kwargs["inferSchema"] = True
    backend = native_namespace(constructor)
    assert_equal_lazy(nw.scan_parquet(parquet_path, backend=backend, **kwargs))


@skipif_pandas_lt_1_5
def test_scan_parquet_kwargs(parquet_path: FileSource) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    assert_equal_lazy(nw.scan_parquet(parquet_path, backend=pd, engine="pyarrow"))


@spark_like_backend
@pytest.mark.parametrize("scan_method", ["scan_csv", "scan_parquet"])
def test_scan_fail_spark_like_without_session(
    backend: _SparkLike, scan_method: str
) -> None:
    pytest.importorskip(backend)
    pattern = re.compile(r"spark.+backend.+require.+session", re.IGNORECASE)
    with pytest.raises(ValueError, match=pattern):
        getattr(nw, scan_method)("unused.csv", backend=backend)


def _pyarrow_parse_options() -> dict[str, Any]:
    from pyarrow import csv

    return {"parse_options": csv.ParseOptions(delimiter=";")}


@pytest.mark.parametrize(
    ("backend", "into_kwargs"),
    [("pyarrow", _pyarrow_parse_options), ("pandas", lambda: {"sep": ";"})],
)
def test_read_csv_raise_on_conflicting_separator(
    backend: Literal["pandas", "pyarrow"], into_kwargs: Callable[[], dict[str, Any]]
) -> None:
    pytest.importorskip(backend)
    kwargs = into_kwargs()
    with pytest.raises(TypeError, match="do not match:"):
        nw.read_csv("unused.csv", backend=backend, separator="|", **kwargs)


@pytest.mark.parametrize(
    ("backend", "into_kwargs"),
    [
        ("pyarrow", _pyarrow_parse_options),
        ("pandas", lambda: {"sep": ";"}),
        ("duckdb", lambda: {"delimiter": ";"}),
        ("duckdb", lambda: {"delim": ";"}),
        ("sqlframe", lambda: {"sep": ";"}),
        ("sqlframe", lambda: {"delimiter": ";"}),
    ],
)
def test_scan_csv_raise_on_conflicting_separator(
    backend: Literal["duckdb", "pandas", "pyarrow", "sqlframe"],
    into_kwargs: Callable[[], dict[str, Any]],
) -> None:
    # Separator validation raises before the source is read (and, for spark-like
    # backends, before a session is required), so a literal path and a string
    # backend are enough. Backend objects are imported lazily via `into_kwargs`.
    pytest.importorskip(backend)
    kwargs = into_kwargs()
    with pytest.raises(TypeError, match="do not match:"):
        nw.scan_csv("unused.csv", backend=backend, separator="|", **kwargs)


def _csv_buffer(
    into: type[StringIO | BytesIO] = BytesIO, /, *, separator: str = ","
) -> StringIO | BytesIO:
    # Polars `write_csv` emits bytes. Older versions raise if that lands in
    # a text buffer (`OSError: string argument expected, got 'bytes'`).
    raw = BytesIO()
    pl.DataFrame(data).write_csv(raw, separator=separator)
    payload = raw.getvalue()
    if into is StringIO:
        return StringIO(payload.decode())
    return BytesIO(payload)


def _parquet_buffer() -> BytesIO:
    buf = BytesIO()
    pl.DataFrame(data).write_parquet(buf)
    buf.seek(0)
    return buf


@pytest.mark.parametrize("into", [StringIO, BytesIO])
def test_read_csv_file_like(
    eager_backend: EagerAllowed, into: type[StringIO | BytesIO]
) -> None:
    assert_equal_eager(nw.read_csv(_csv_buffer(into), backend=eager_backend))
    assert_equal_eager(
        nw.read_csv(
            _csv_buffer(into, separator="|"), backend=eager_backend, separator="|"
        )
    )


@pytest.mark.parametrize("into", [StringIO, BytesIO])
def test_scan_csv_file_like(
    eager_backend: EagerAllowed, into: type[StringIO | BytesIO]
) -> None:
    assert_equal_lazy(nw.scan_csv(_csv_buffer(into), backend=eager_backend))


def test_read_csv_file_like_pyarrow_encoding() -> None:
    # A text buffer is re-encoded before reaching `pyarrow.csv`, so it has to use
    # whatever `read_options.encoding` declares, else the values come back mojibake.
    from pyarrow import csv

    result = nw.read_csv(
        StringIO("z\nrené\n"),
        backend="pyarrow",
        read_options=csv.ReadOptions(encoding="latin-1"),
    )
    assert result.rows() == [("rené",)]


@skipif_pandas_lt_1_5
def test_read_parquet_file_like(eager_backend: EagerAllowed) -> None:
    assert_equal_eager(nw.read_parquet(_parquet_buffer(), backend=eager_backend))


@skipif_pandas_lt_1_5
def test_scan_parquet_file_like(eager_backend: EagerAllowed) -> None:
    assert_equal_lazy(nw.scan_parquet(_parquet_buffer(), backend=eager_backend))


@path_only_backend
def test_scan_csv_file_like_unsupported(backend: _LazyOnly) -> None:
    pytest.importorskip(backend)
    with pytest.raises(TypeError, match="file-like"):
        nw.scan_csv(_csv_buffer(StringIO), backend=backend, **_session_kwargs(backend))


@path_only_backend
def test_scan_parquet_file_like_unsupported(backend: _LazyOnly) -> None:
    pytest.importorskip(backend)
    with pytest.raises(TypeError, match="file-like"):
        nw.scan_parquet(_parquet_buffer(), backend=backend, **_session_kwargs(backend))


@pytest.mark.parametrize("into", [StringIO, BytesIO])
def test_scan_csv_file_like_duckdb(into: type[StringIO | BytesIO]) -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("fsspec")
    assert_equal_lazy(nw.scan_csv(_csv_buffer(into), backend="duckdb"))


def test_scan_parquet_file_like_duckdb() -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("fsspec")
    context = (
        pytest.raises(NotImplementedError, match=r"duckdb>=1\.5\.4")
        if DUCKDB_VERSION < (1, 5, 4)
        else does_not_raise()
    )
    with context:
        assert_equal_lazy(nw.scan_parquet(_parquet_buffer(), backend="duckdb"))
