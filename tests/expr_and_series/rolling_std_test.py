from __future__ import annotations

import numpy as np
import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0]}

kwargs_and_expected = {
    "x1": {
        "kwargs": {"window_size": 3},
        "expected": np.sqrt([float("nan"), float("nan"), 1 / 3, 1, 4 / 3, 7 / 3, 3]),
    },
    "x2": {
        "kwargs": {"window_size": 3, "min_periods": 1},
        "expected": np.sqrt([float("nan"), 0.5, 1 / 3, 1.0, 4 / 3, 7 / 3, 3]),
    },
    "x3": {
        "kwargs": {"window_size": 2, "min_periods": 1},
        "expected": np.sqrt([float("nan"), 0.5, 0.5, 2.0, 2.0, 4.5, 4.5]),
    },
    "x4": {
        "kwargs": {"window_size": 5, "min_periods": 1, "center": True},
        "expected": np.sqrt([1 / 3, 11 / 12, 4 / 5, 17 / 10, 2.0, 2.25, 3]),
    },
    "x5": {
        "kwargs": {"window_size": 4, "min_periods": 1, "center": True},
        "expected": np.sqrt([0.5, 1 / 3, 11 / 12, 11 / 12, 2.25, 2.25, 3]),
    },
    "x6": {
        "kwargs": {"window_size": 3, "ddof": 2},
        "expected": np.sqrt([float("nan"), float("nan"), 2 / 3, 2.0, 8 / 3, 14 / 3, 6.0]),
    },
}


@pytest.mark.filterwarnings(
    "ignore:`Expr.rolling_std` is being called from the stable API although considered an unstable feature."
)
def test_rolling_std_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "dask" in str(constructor):
        # TODO(FBruzzesi): Dask is raising the following error:
        # NotImplementedError: Partition size is less than overlapping window size.
        # Try using ``df.repartition`` to increase the partition size.
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(
        **{
            name: nw.col("a").rolling_std(**values["kwargs"])
            for name, values in kwargs_and_expected.items()
        }
    )
    expected = {name: values["expected"] for name, values in kwargs_and_expected.items()}

    assert_equal_data(result, expected)


@pytest.mark.filterwarnings(
    "ignore:`Series.rolling_std` is being called from the stable API although considered an unstable feature."
)
def test_rolling_std_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result = df.select(
        **{
            name: df["a"].rolling_std(**values["kwargs"])
            for name, values in kwargs_and_expected.items()
        }
    )
    expected = {name: values["expected"] for name, values in kwargs_and_expected.items()}
    assert_equal_data(result, expected)
