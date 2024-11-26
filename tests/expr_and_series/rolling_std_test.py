from __future__ import annotations

import random

import hypothesis.strategies as st
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [None, 1, 2, None, 4, 6, 11]}

kwargs_and_expected = {
    "x1": {"kwargs": {"window_size": 3}, "expected": [float("nan")] * 6 + [7.0]},
    "x2": {
        "kwargs": {"window_size": 3, "min_periods": 1},
        "expected": [float("nan"), 1.0, 1.5, 1.5, 3.0, 5.0, 7.0],
    },
    "x3": {
        "kwargs": {"window_size": 2, "min_periods": 1},
        "expected": [float("nan"), 1.0, 1.5, 2.0, 4.0, 5.0, 8.5],
    },
    "x4": {
        "kwargs": {"window_size": 5, "min_periods": 1, "center": True},
        "expected": [1.5, 1.5, 7 / 3, 3.25, 5.75, 7.0, 7.0],
    },
    "x5": {
        "kwargs": {"window_size": 4, "min_periods": 1, "center": True},
        "expected": [1.0, 1.5, 1.5, 7 / 3, 4.0, 7.0, 7.0],
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
            name: nw.col("a").rolling_std(**values["kwargs"])  # type: ignore[arg-type]
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
            name: df["a"].rolling_std(**values["kwargs"])  # type: ignore[arg-type]
            for name, values in kwargs_and_expected.items()
        }
    )
    expected = {name: values["expected"] for name, values in kwargs_and_expected.items()}
    assert_equal_data(result, expected)


@given(  # type: ignore[misc]
    center=st.booleans(),
    values=st.lists(st.floats(-10, 10), min_size=3, max_size=10),
)
@pytest.mark.skipif(PANDAS_VERSION < (1,), reason="too old for pyarrow")
@pytest.mark.filterwarnings("ignore:.*:narwhals.exceptions.NarwhalsUnstableWarning")
def test_rolling_std_hypothesis(center: bool, values: list[float]) -> None:  # noqa: FBT001
    s = pd.Series(values)
    n_missing = random.randint(0, len(s) - 1)  # noqa: S311
    window_size = random.randint(1, len(s))  # noqa: S311
    min_periods = random.randint(1, window_size)  # noqa: S311
    mask = random.sample(range(len(s)), n_missing)
    s[mask] = None
    df = pd.DataFrame({"a": s})
    expected = (
        s.rolling(window=window_size, center=center, min_periods=min_periods)
        .mean()
        .to_frame("a")
    )
    result = nw.from_native(pa.Table.from_pandas(df)).select(
        nw.col("a").rolling_std(window_size, center=center, min_periods=min_periods)
    )
    expected_dict = nw.from_native(expected, eager_only=True).to_dict(as_series=False)
    assert_equal_data(result, expected_dict)
