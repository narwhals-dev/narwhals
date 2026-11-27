from __future__ import annotations

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    Constructor,
    assert_equal_data,
    xfail_if_builtin_dask,
)


def test_meanh(constructor: Constructor) -> None:
    data = {"a": [1, 3, None, None], "b": [4, None, 6, None]}
    df = nw.from_native(constructor(data))
    mean = nw.mean_horizontal(nw.col("a"), nw.col("b"))
    # `assert_equal_data` treats NaN and None as equal, so check `is_null` explicitly
    result = df.select(horizontal_mean=mean, is_null=mean.is_null())
    expected = {
        "horizontal_mean": [2.5, 3.0, 6.0, None],
        "is_null": [False, False, False, True],
    }
    assert_equal_data(result, expected)


def test_meanh_with_literal(constructor: Constructor) -> None:
    data = {"a": [1, 3, None, None], "b": [4, None, 6, None]}
    df = nw.from_native(constructor(data))
    result = df.select(horizontal_mean=nw.mean_horizontal(nw.lit(1), "a", nw.col("b")))
    expected = {"horizontal_mean": [2.0, 2.0, 3.5, 1.0]}
    assert_equal_data(result, expected)


def test_arrow_horizontal_mean_nulls(
    arrow_constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    df = nw.from_native(arrow_constructor({"a": [None, 3], "b": [None, 7]}))
    if PANDAS_VERSION < PkgVersion("3"):
        xfail_if_builtin_dask(df, request)
    mean = nw.mean_horizontal("a", "b")
    assert_equal_data(
        df.select(mean=mean, null=mean.is_null()),
        {"mean": [None, 5.0], "null": [True, False]},
    )


def test_dask_horizontal_mean_with_typed_literal() -> None:
    dd = pytest.importorskip("dask.dataframe")
    pytest.importorskip("pyarrow")
    import pandas as pd

    frame = pd.DataFrame(
        {"a": pd.array([1, 2, 3], dtype="int64[pyarrow]")}, index=[9, 4, 8]
    )
    df = nw.from_native(dd.from_pandas(frame, npartitions=2, sort=False))
    result = (
        df.select(mean=nw.mean_horizontal("a", nw.lit(2, nw.Int64))).to_native().compute()
    )
    assert result.index.equals(frame.index)
    assert result["mean"].tolist() == [1.5, 2.0, 2.5]


def test_meanh_all(constructor: Constructor) -> None:
    data = {"a": [2, 4, 6], "b": [10, 20, 30]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.mean_horizontal(nw.all()))
    expected = {"a": [6, 12, 18]}
    assert_equal_data(result, expected)
    result = df.select(c=nw.mean_horizontal(nw.all()))
    expected = {"c": [6, 12, 18]}
    assert_equal_data(result, expected)
