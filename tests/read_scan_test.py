from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    Constructor,
    assert_equal_data,
    is_windows,
    pyspark_session,
    sqlframe_session,
)

pytest.importorskip("polars")
pytest.importorskip("pyarrow")
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from types import ModuleType

    from typing_extensions import TypeAlias

    from narwhals._typing import EagerAllowed, _LazyOnly, _SparkLike
    from narwhals.typing import FileSource

IOSourceKind: TypeAlias = Literal["str", "Path", "PathLike"]

data: Mapping[str, Any] = {"a": [1, 2, 3], "b": [4.5, 6.7, 8.9], "z": ["x", "y", "w"]}
skipif_pandas_lt_1_5 = pytest.mark.skipif(
    PANDAS_VERSION < (1, 5), reason="too old for pyarrow"
)
lazy_core_backend = pytest.mark.parametrize("backend", ["duckdb", "ibis", "sqlframe"])
spark_like_backend = pytest.mark.parametrize("backend", ["pyspark", "sqlframe"])


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


@pytest.fixture(scope="module", params=["str", "Path", "PathLike"])
def csv_path(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> FileSource:
    fp = tmp_path_factory.mktemp("data") / "file.csv"
    pl.DataFrame(data).write_csv(fp)
    return _into_file_source(fp, request.param)


@pytest.fixture(scope="module", params=["str", "Path", "PathLike"])
def parquet_path(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> FileSource:
    fp = tmp_path_factory.mktemp("data") / "file.parquet"
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


def test_read_csv(csv_path: FileSource, eager_backend: EagerAllowed) -> None:
    assert_equal_eager(nw.read_csv(csv_path, backend=eager_backend))


@skipif_pandas_lt_1_5
def test_read_csv_kwargs(csv_path: FileSource) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    assert_equal_eager(nw.read_csv(csv_path, backend=pd, engine="pyarrow"))


@lazy_core_backend
def test_read_csv_raise_with_lazy(backend: _LazyOnly) -> None:
    pytest.importorskip(backend)
    with pytest.raises(ValueError, match="Expected eager backend, found"):
        nw.read_csv("unused.csv", backend=backend)  # type: ignore[arg-type]


def test_scan_csv(csv_path: FileSource, constructor: Constructor) -> None:
    kwargs: dict[str, Any]
    if "sqlframe" in str(constructor):
        kwargs = {"session": sqlframe_session(), "inferSchema": True, "header": True}
    elif "pyspark" in str(constructor):
        kwargs = {"session": pyspark_session(), "inferSchema": True, "header": True}
    else:
        kwargs = {}
    backend = native_namespace(constructor)
    assert_equal_lazy(nw.scan_csv(csv_path, backend=backend, **kwargs))


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
    kwargs: dict[str, Any]
    if "sqlframe" in str(constructor) and is_windows():
        reason = "_duckdb.IOException: IO Error: No files found that match the pattern"
        pytest.skip(reason)

    if "sqlframe" in str(constructor):
        kwargs = {"session": sqlframe_session(), "inferSchema": True}
    elif "pyspark" in str(constructor):
        kwargs = {"session": pyspark_session(), "inferSchema": True, "header": True}
    else:
        kwargs = {}
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
