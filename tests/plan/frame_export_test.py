from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest

from tests.plan.utils import dataframe
from tests.utils import is_windows

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import TypeAlias

    from narwhals.typing import FileSource
    from tests.conftest import Data

pytest.importorskip("pyarrow")

IOTargetKind: TypeAlias = Literal["str", "Path", "PathLike"]
"""Duplicated from `tests.read_scan_test.py`.

Needs extending for `BytesIO`.
"""


class MockPathLike:
    def __init__(self, path: Path) -> None:
        self._super_secret: Path = path

    def __fspath__(self) -> str:
        return self._super_secret.__fspath__()


def _into_file_source(source: Path, which: IOTargetKind, /) -> FileSource:
    mapping: Mapping[IOTargetKind, FileSource] = {
        "str": str(source),
        "Path": source,
        "PathLike": MockPathLike(source),
    }
    return mapping[which]


@pytest.fixture(params=["str", "Path", "PathLike"])
def csv_path(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> FileSource:
    fp = tmp_path_factory.mktemp("data") / "file.csv"
    return _into_file_source(fp, request.param)


@pytest.fixture(params=["str", "Path", "PathLike"])
def parquet_path(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> FileSource:
    fp = tmp_path_factory.mktemp("data") / "file.parquet"
    return _into_file_source(fp, request.param)


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [1, 2, 3]}


XFAIL_DATAFRAME_EXPORT = pytest.mark.xfail(
    reason="TODO: `DataFrame.write_{csv,parquet}`()", raises=NotImplementedError
)


@XFAIL_DATAFRAME_EXPORT
def test_write_csv(data: Data, csv_path: FileSource) -> None:  # pragma: no cover
    df = dataframe(data)
    result_none = df.write_csv(csv_path)
    assert Path(csv_path).exists()
    assert result_none is None
    result = dataframe(data).write_csv()
    if is_windows():  # pragma: no cover
        result = result.replace("\r\n", "\n")
    if df.implementation.is_pyarrow():
        assert result == '"a"\n1\n2\n3\n'
    else:  # pragma: no cover
        assert result == "a\n1\n2\n3\n"


@XFAIL_DATAFRAME_EXPORT
def test_write_parquet(data: Data, parquet_path: FileSource) -> None:  # pragma: no cover
    dataframe(data).write_parquet(parquet_path)
    assert Path(parquet_path).exists()


@pytest.mark.xfail(
    reason="TODO: `DataFrame.lazy()`, `LazyFrame.sink_parquet()`", raises=AttributeError
)
def test_sink_parquet(data: Data, parquet_path: FileSource) -> None:  # pragma: no cover
    df = dataframe(data)
    df.lazy().sink_parquet(parquet_path)  # type: ignore[attr-defined]
    assert Path(parquet_path).exists()
