from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 1, 2], "b": [1, 2, 3]}


def test_ewm_mean_expr(constructor: Constructor) -> None:
    if "pyarrow_" in str(constructor) or "dask" in str(constructor):  # remove
        pytest.skip()

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b").ewm_mean(com=1))
    expected = {
        "a": [1.0, 1.0, 1.5714285714285714],
        "b": [1.0, 1.6666666666666667, 2.4285714285714284],
    }
    assert_equal_data(result, expected)


def test_ewm_mean_series(constructor_eager: ConstructorEager) -> None:
    if "pyarrow_" in str(constructor_eager):  # remove
        pytest.skip()

    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.ewm_mean(com=1)
    expected = {"a": [1.0, 1.0, 1.5714285714285714]}
    assert_equal_data({"a": result}, expected)


@pytest.mark.parametrize("adjust", [True, False])
def test_ewm_mean_expr_adjust(
    constructor: Constructor,
    adjust: bool,  # noqa: FBT001
) -> None:
    if "pyarrow_" in str(constructor) or "dask" in str(constructor):  # remove
        pytest.skip()

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b").ewm_mean(com=1, adjust=adjust))

    if adjust:
        expected = {
            "a": [1.0, 1.0, 1.5714285714285714],
            "b": [1.0, 1.6666666666666667, 2.4285714285714284],
        }
    else:
        expected = {
            "a": [1.0, 1.0, 1.5],
            "b": [1.0, 1.5, 2.25],
        }

    assert_equal_data(result, expected)
