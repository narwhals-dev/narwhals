from __future__ import annotations

from math import sqrt
from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import POLARS_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0]}

kwargs_and_expected = (
    {
        "name": "x1",
        "kwargs": {"window_size": 3},
        "expected": [
            sqrt(x) if x is not None else x
            for x in [None, None, 1 / 3, 1, 4 / 3, 7 / 3, 3]
        ],
    },
    {
        "name": "x2",
        "kwargs": {"window_size": 3, "min_periods": 1},
        "expected": [
            sqrt(x) if x is not None else x
            for x in [None, 0.5, 1 / 3, 1.0, 4 / 3, 7 / 3, 3]
        ],
    },
    {
        "name": "x3",
        "kwargs": {"window_size": 2, "min_periods": 1},
        "expected": [
            sqrt(x) if x is not None else x for x in [None, 0.5, 0.5, 2.0, 2.0, 4.5, 4.5]
        ],
    },
    {
        "name": "x4",
        "kwargs": {"window_size": 5, "min_periods": 1, "center": True},
        "expected": [
            sqrt(x) if x is not None else x
            for x in [1 / 3, 11 / 12, 4 / 5, 17 / 10, 2.0, 2.25, 3]
        ],
    },
    {
        "name": "x5",
        "kwargs": {"window_size": 4, "min_periods": 1, "center": True},
        "expected": [
            sqrt(x) if x is not None else x
            for x in [0.5, 1 / 3, 11 / 12, 11 / 12, 2.25, 2.25, 3]
        ],
    },
    {
        "name": "x6",
        "kwargs": {"window_size": 3, "ddof": 2},
        "expected": [
            sqrt(x) if x is not None else x
            for x in [None, None, 2 / 3, 2.0, 8 / 3, 14 / 3, 6.0]
        ],
    },
)


@pytest.mark.filterwarnings(
    "ignore:`Expr.rolling_std` is being called from the stable API although considered an unstable feature."
)
@pytest.mark.parametrize("kwargs_and_expected", kwargs_and_expected)
def test_rolling_std_expr(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    kwargs_and_expected: dict[str, Any],
) -> None:
    name = kwargs_and_expected["name"]
    kwargs = kwargs_and_expected["kwargs"]
    expected = kwargs_and_expected["expected"]

    if (
        "dask" in str(constructor)
        or ("polars" in str(constructor) and POLARS_VERSION < (1,))
        or "duckdb" in str(constructor)
    ):
        # TODO(FBruzzesi): Dask is raising the following error:
        # NotImplementedError: Partition size is less than overlapping window size.
        # Try using ``df.repartition`` to increase the partition size.
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").rolling_std(**kwargs).alias(name))

    assert_equal_data(result, {name: expected})


@pytest.mark.filterwarnings(
    "ignore:`Series.rolling_std` is being called from the stable API although considered an unstable feature."
)
@pytest.mark.parametrize("kwargs_and_expected", kwargs_and_expected)
def test_rolling_std_series(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    kwargs_and_expected: dict[str, Any],
) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1,):
        request.applymarker(pytest.mark.xfail)

    name = kwargs_and_expected["name"]
    kwargs = kwargs_and_expected["kwargs"]
    expected = kwargs_and_expected["expected"]

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].rolling_std(**kwargs).alias(name))

    assert_equal_data(result, {name: expected})
