from __future__ import annotations

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": [-1, 0, 1, 2, 4]}

expected = [float("nan"), 0, 1, 1.414213, 2]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_sqrt_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").sqrt())
    assert_equal_data(result, {"a": expected})


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_sqrt_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.sqrt()
    assert_equal_data({"a": result}, {"a": expected})


def test_sqrt_dtype_pandas() -> None:
    s = pd.Series([1.0, 2.0], name="a", dtype="float32", index=[8, 7])
    result = nw.from_native(s, series_only=True).sqrt().to_native()
    expected = pd.Series([1, 1.414213], name="a", dtype="float32", index=[8, 7])
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="nullable types require pandas2+")
def test_sqrt_dtype_pandas_nullable() -> None:
    s = pd.Series([1.0, None, 2.0], name="a", dtype="Float32", index=[8, 7, 6])
    result = nw.from_native(s, series_only=True).sqrt().to_native()
    expected = pd.Series([1, None, 1.414213], name="a", dtype="Float32", index=[8, 7, 6])
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(PANDAS_VERSION < (2, 1, 0), reason="nullable types require pandas2+")
def test_sqrt_dtype_pandas_pyarrow() -> None:
    s = pd.Series([1.0, None, 2.0], name="a", dtype="Float32[pyarrow]", index=[8, 7, 6])
    result = nw.from_native(s, series_only=True).sqrt().to_native()
    expected = pd.Series(
        [1, None, 1.414213], name="a", dtype="Float64[pyarrow]", index=[8, 7, 6]
    )
    pd.testing.assert_series_equal(result, expected)
