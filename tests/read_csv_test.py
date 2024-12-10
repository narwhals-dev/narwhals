from __future__ import annotations

import polars as pl
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"a": [1, 2, 3], "b": [4.5, 6.7, 8.9], "z": ["x", "y", "w"]}


def test_read_csv(
    tmpdir: pytest.TempdirFactory,
    constructor: Constructor,
    request: pytest.FixtureRequest,
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    df_pl.write_csv(filepath)
    df = nw.from_native(constructor(data))
    native_namespace = nw.get_native_namespace(df)
    result = nw.read_csv(filepath, native_namespace=native_namespace)
    assert_equal_data(result, data)


def test_read_csv_v1(
    tmpdir: pytest.TempdirFactory,
    constructor: Constructor,
    request: pytest.FixtureRequest,
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    df_pl.write_csv(filepath)
    df = nw_v1.from_native(constructor(data))
    native_namespace = nw_v1.get_native_namespace(df)
    result = nw_v1.read_csv(filepath, native_namespace=native_namespace)
    assert_equal_data(result, data)
