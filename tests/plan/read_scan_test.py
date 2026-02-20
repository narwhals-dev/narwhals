from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data
from tests.utils import PANDAS_VERSION

pytest.importorskip("polars")
pytest.importorskip("pyarrow")
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from typing_extensions import TypeAlias

    from narwhals._typing import EagerAllowed, _LazyOnly
    from narwhals.typing import FileSource
    from tests.conftest import Data

IOSourceKind: TypeAlias = Literal["str", "Path", "PathLike"]
IntoKwds: TypeAlias = "dict[str, Any] | Callable[[], dict[str, Any]]"
"""Keyword-arguments, or a callback that returns them."""


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [1, 2, 3], "b": [4.5, 6.7, 8.9], "z": ["x", "y", "w"]}


lazy_core_backend = pytest.mark.parametrize("backend", ["duckdb", "ibis", "sqlframe"])
param_pandas_import = pytest.param(
    "pandas",
    {"engine": "pyarrow"},
    marks=[
        pytest.mark.xfail(reason="Not implemented pandas", raises=NotImplementedError),
        pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow"),
    ],
)


def pyarrow_read_csv_kwds() -> dict[str, Any]:
    from pyarrow import csv

    return {
        "read_options": csv.ReadOptions(column_names=None),
        "parse_options": csv.ParseOptions(delimiter=","),
        "convert_options": csv.ConvertOptions(include_columns=["a", "b", "z"]),
    }


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


def _into_kwds(into_kwds: IntoKwds) -> dict[str, Any]:
    return into_kwds if not callable(into_kwds) else into_kwds()


@pytest.fixture(scope="module", params=["str", "Path", "PathLike"])
def csv_path(
    data: Data, tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> FileSource:
    fp = tmp_path_factory.mktemp("data") / "file.csv"
    pl.DataFrame(data).write_csv(fp)
    return _into_file_source(fp, request.param)


@pytest.fixture(scope="module", params=["str", "Path", "PathLike"])
def parquet_path(
    data: Data, tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> FileSource:
    fp = tmp_path_factory.mktemp("data") / "file.parquet"
    pl.DataFrame(data).write_parquet(fp)
    return _into_file_source(fp, request.param)


def assert_equal_eager(result: nwp.DataFrame[Any], expected: Data) -> None:
    assert_equal_data(result, expected)
    assert isinstance(result, nwp.DataFrame)


def test_read_csv(data: Data, csv_path: FileSource, eager: EagerAllowed) -> None:
    assert_equal_eager(nwp.read_csv(csv_path, backend=eager), data)


@pytest.mark.parametrize(
    ("backend", "into_kwds"), [param_pandas_import, ("pyarrow", pyarrow_read_csv_kwds)]
)
def test_read_csv_kwargs(
    data: Data, csv_path: FileSource, backend: EagerAllowed, into_kwds: IntoKwds
) -> None:
    kwds = _into_kwds(into_kwds)
    assert_equal_eager(nwp.read_csv(csv_path, backend=backend, **kwds), data)


@lazy_core_backend
def test_read_csv_raise_with_lazy(backend: _LazyOnly) -> None:
    pytest.importorskip(backend)
    with pytest.raises(ValueError, match="support in Narwhals is lazy-only"):
        nwp.read_csv("unused.csv", backend=backend)  # type: ignore[call-overload]


def test_read_parquet(data: Data, parquet_path: FileSource, eager: EagerAllowed) -> None:
    assert_equal_eager(nwp.read_parquet(parquet_path, backend=eager), data)


@pytest.mark.parametrize(
    ("backend", "into_kwds"),
    [
        param_pandas_import,
        ("pyarrow", {"use_threads": False, "columns": ["a", "b", "z"]}),
    ],
)
def test_read_parquet_kwargs(
    data: Data, parquet_path: FileSource, backend: EagerAllowed, into_kwds: IntoKwds
) -> None:
    kwds = _into_kwds(into_kwds)
    assert_equal_eager(nwp.read_parquet(parquet_path, backend=backend, **kwds), data)


@lazy_core_backend
def test_read_parquet_raise_with_lazy(backend: _LazyOnly) -> None:
    pytest.importorskip(backend)
    with pytest.raises(ValueError, match="support in Narwhals is lazy-only"):
        nwp.read_parquet("unused.parquet", backend=backend)  # type: ignore[call-overload]
