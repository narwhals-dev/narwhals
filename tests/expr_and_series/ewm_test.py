from __future__ import annotations

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from tests.utils import POLARS_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 1, 2], "b": [1, 2, 3]}


def test_ewm_mean_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table_", "dask", "modin")) or (
        "polars" in str(constructor) and POLARS_VERSION <= (0, 20, 31)
    ):
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
    if any(x in str(constructor_eager) for x in ("pyarrow_table_", "modin")) or (
        "polars" in str(constructor_eager) and POLARS_VERSION <= (0, 20, 31)
    ):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.ewm_mean(com=1)
    expected = {"a": [1.0, 1.0, 1.5714285714285714]}
    assert_equal_data({"a": result}, expected)


@pytest.mark.parametrize(
    ("adjust", "expected"),
    [
        (
            True,
            {
                "a": [1.0, 1.0, 1.5714285714285714],
                "b": [1.0, 1.6666666666666667, 2.4285714285714284],
            },
        ),
        (
            False,
            {
                "a": [1.0, 1.0, 1.5],
                "b": [1.0, 1.5, 2.25],
            },
        ),
    ],
)
def test_ewm_mean_expr_adjust(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    adjust: bool,  # noqa: FBT001
    expected: dict[str, list[float]],
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table_", "dask", "modin")) or (
        "polars" in str(constructor) and POLARS_VERSION <= (0, 20, 31)
    ):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b").ewm_mean(com=1, adjust=adjust))
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


@pytest.mark.parametrize(
    ("ignore_nulls", "expected"),
    [
        (
            True,
            {
                "a": [
                    2.0,
                    3.3333333333333335,
                    float("nan"),
                    3.142857142857143,
                ]
            },
        ),
        (
            False,
            {
                "a": [
                    2.0,
                    3.3333333333333335,
                    float("nan"),
                    3.090909090909091,
                ]
            },
        ),
    ],
)
def test_ewm_mean_nulls(
    request: pytest.FixtureRequest,
    ignore_nulls: bool,  # noqa: FBT001
    expected: dict[str, list[float]],
    constructor: Constructor,
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table_", "dask", "modin")) or (
        "polars" in str(constructor) and POLARS_VERSION <= (0, 20, 31)
    ):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor({"a": [2.0, 4.0, None, 3.0]}))
    result = df.select(nw.col("a").ewm_mean(com=1, ignore_nulls=ignore_nulls))
    assert_equal_data(result, expected)


def test_ewm_mean_params(
    request: pytest.FixtureRequest,
    constructor: Constructor,
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table_", "dask", "modin")) or (
        "polars" in str(constructor) and POLARS_VERSION <= (0, 20, 31)
    ):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor({"a": [2, 5, 3]}))
    expected = {"a": [2.0, 4.0, 3.4285714285714284]}
    assert_equal_data(
        df.select(nw.col("a").ewm_mean(alpha=0.5, adjust=True, ignore_nulls=True)),
        expected,
    )

    expected = {"a": [2.0, 4.500000000000001, 3.2903225806451615]}
    assert_equal_data(
        df.select(nw.col("a").ewm_mean(span=1.5, adjust=True, ignore_nulls=True)),
        expected,
    )

    expected = {"a": [2.0, 3.1101184251576903, 3.0693702609187237]}
    assert_equal_data(
        df.select(nw.col("a").ewm_mean(half_life=1.5, adjust=False)), expected
    )

    expected = {"a": [float("nan"), 4.0, 3.4285714285714284]}
    assert_equal_data(
        df.select(
            nw.col("a").ewm_mean(alpha=0.5, adjust=True, min_periods=2, ignore_nulls=True)
        ),
        expected,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        df.select(nw.col("a").ewm_mean(span=1.5, half_life=0.75, ignore_nulls=False))
