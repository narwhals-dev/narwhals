from __future__ import annotations

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 1, 2], "b": [1, 2, 3]}


def test_ewm_mean_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table_", "dask")):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b").ewm_mean(com=1))
    expected = {
        "a": [1.0, 1.0, 1.5714285714285714],
        "b": [1.0, 1.6666666666666667, 2.4285714285714284],
    }
    assert_equal_data(result, expected)


def test_ewm_mean_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table_" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.ewm_mean(com=1)
    expected = {"a": [1.0, 1.0, 1.5714285714285714]}
    assert_equal_data({"a": result}, expected)


@pytest.mark.parametrize("adjust", [True, False])
def test_ewm_mean_expr_adjust(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    adjust: bool,  # noqa: FBT001
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table_", "dask")):
        request.applymarker(pytest.mark.xfail)

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


def test_ewm_mean_dask_raise() -> None:
    pytest.importorskip("dask")
    pytest.importorskip("dask_expr", exc_type=ImportError)
    import dask.dataframe as dd

    df = nw.from_native(dd.from_pandas(pd.DataFrame({"a": [1, 2, 3]})))
    with pytest.raises(
        NotImplementedError,
        match="`Expr.ewm_mean` is not supported for the Dask backend",
    ):
        df.select(nw.col("a").ewm_mean(com=1))


@pytest.mark.parametrize("ignore_nulls", [True, False])
def test_ewm_mean_nulls(
    request: pytest.FixtureRequest,
    ignore_nulls: bool,  # noqa: FBT001
    constructor: Constructor,
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table_", "dask")):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor({"a": [2.0, 4.0, None, 3.0]}))
    result = df.select(nw.col("a").ewm_mean(com=1, ignore_nulls=ignore_nulls))

    if ignore_nulls:
        expected = {
            "a": [
                2.0,
                3.3333333333333335,
                float("nan"),
                3.142857142857143,
            ],
        }
    else:
        expected = {
            "a": [
                2.0,
                3.3333333333333335,
                float("nan"),
                3.090909090909091,
            ],
        }

    assert_equal_data(result, expected)
