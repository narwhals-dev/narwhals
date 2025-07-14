from __future__ import annotations

import math

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": [-1, 0, 1, 2, 4]}

expected = {  # base: expected values
    2: [float("nan"), float("-inf"), 0, 1, 2],
    10: [float("nan"), float("-inf"), 0, 0.30103, 0.60206],
    math.e: [float("nan"), float("-inf"), 0, 0.693147, 1.386294],
}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("base", [2, 10, math.e])
def test_log_expr(constructor: Constructor, base: float) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").log(base=base))
    assert_equal_data(result, {"a": expected[base]})


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("base", [2, 10, math.e])
def test_log_series(constructor_eager: ConstructorEager, base: float) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.log(base=base)
    assert_equal_data({"a": result}, {"a": expected[base]})


def test_log_dtype_pandas() -> None:
    s = pd.Series([1.0, 2.0], name="a", dtype="float32", index=[8, 7])
    result = nw.from_native(s, series_only=True).log().to_native()
    expected = pd.Series([0.0, 0.693147], name="a", dtype="float32", index=[8, 7])
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="nullable types require pandas2+")
def test_log_dtype_pandas_nullabe() -> None:
    s = pd.Series([1.0, None, 2.0], name="a", dtype="Float32", index=[8, 7, 6])
    result = nw.from_native(s, series_only=True).log().to_native()
    expected = pd.Series(
        [0.0, None, 0.693147], name="a", dtype="Float32", index=[8, 7, 6]
    )
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(PANDAS_VERSION < (2, 1, 0), reason="nullable types require pandas2+")
def test_log_dtype_pandas_pyarrow() -> None:
    s = pd.Series([1.0, None, 2.0], name="a", dtype="Float32[pyarrow]", index=[8, 7, 6])
    result = nw.from_native(s, series_only=True).log().to_native()
    expected = pd.Series(
        [0.0, None, 0.693147], name="a", dtype="Float64[pyarrow]", index=[8, 7, 6]
    )
    pd.testing.assert_series_equal(result, expected)
