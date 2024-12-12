from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import PANDAS_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"a": [1, 2, 3], "b": [4.5, 6.7, 8.9], "z": ["x", "y", "w"]}


def test_scan_csv(
    tmpdir: pytest.TempdirFactory,
    constructor: Constructor,
) -> None:
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    df_pl.write_csv(filepath)
    df = nw.from_native(constructor(data))
    native_namespace = nw.get_native_namespace(df)
    result = nw.scan_csv(filepath, native_namespace=native_namespace)
    assert_equal_data(result.collect(), data)
    assert isinstance(result, nw.LazyFrame)


def test_scan_csv_v1(
    tmpdir: pytest.TempdirFactory,
    constructor: Constructor,
) -> None:
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    df_pl.write_csv(filepath)
    df = nw_v1.from_native(constructor(data))
    native_namespace = nw_v1.get_native_namespace(df)
    result = nw_v1.scan_csv(filepath, native_namespace=native_namespace)
    assert_equal_data(result.collect(), data)
    assert isinstance(result, nw_v1.LazyFrame)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_scan_csv_kwargs(tmpdir: pytest.TempdirFactory) -> None:
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    df_pl.write_csv(filepath)
    result = nw.scan_csv(filepath, native_namespace=pd, engine="pyarrow")
    assert_equal_data(result, data)
