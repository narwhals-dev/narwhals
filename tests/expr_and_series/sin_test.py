from __future__ import annotations

from math import pi

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": [-pi, -pi / 2, 0, pi / 2, pi]}

expected = [0, -1, 0, 1, 0]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_sin_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").sin())
    assert_equal_data(result, {"a": expected})


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_sin_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.sin()
    assert_equal_data({"a": result}, {"a": expected})


def test_sin_dtype_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s = pd.Series([-pi / 2, 0], name="a", dtype="float32", index=[8, 7])
    result = nw.from_native(s, series_only=True).sin().to_native()
    expected = pd.Series([-1, 0], name="a", dtype="float32", index=[8, 7])
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="nullable types require pandas2+")
def test_sin_dtype_pandas_nullabe() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s = pd.Series([-pi / 2, None, pi / 2], name="a", dtype="Float32", index=[8, 7, 6])
    result = nw.from_native(s, series_only=True).sin().to_native()
    expected = pd.Series([-1, None, 1], name="a", dtype="Float32", index=[8, 7, 6])
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(PANDAS_VERSION < (2, 1, 0), reason="nullable types require pandas2+")
def test_sin_dtype_pandas_pyarrow() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    s = pd.Series(
        [-pi / 2, None, pi / 2], name="a", dtype="Float32[pyarrow]", index=[8, 7, 6]
    )
    result = nw.from_native(s, series_only=True).sin().to_native()
    expected = pd.Series(
        [-1, None, 1], name="a", dtype="Float32[pyarrow]", index=[8, 7, 6]
    )
    pd.testing.assert_series_equal(result, expected)
