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

data = {"a": [1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0]}

kwargs_and_expected = {
    "x1": {
        "kwargs": {"window_size": 3},
        "expected": [float("nan"), float("nan"), 1 / 3, 1, 4 / 3, 7 / 3, 3],
    },
    "x2": {
        "kwargs": {"window_size": 3, "min_periods": 1},
        "expected": [float("nan"), 0.5, 1 / 3, 1.0, 4 / 3, 7 / 3, 3],
    },
    "x3": {
        "kwargs": {"window_size": 2, "min_periods": 1},
        "expected": [float("nan"), 0.5, 0.5, 2.0, 2.0, 4.5, 4.5],
    },
    "x4": {
        "kwargs": {"window_size": 5, "min_periods": 1, "center": True},
        "expected": [1 / 3, 11 / 12, 4 / 5, 17 / 10, 2.0, 2.25, 3],
    },
    "x5": {
        "kwargs": {"window_size": 4, "min_periods": 1, "center": True},
        "expected": [0.5, 1 / 3, 11 / 12, 11 / 12, 2.25, 2.25, 3],
    },
    "x6": {
        "kwargs": {"window_size": 3, "ddof": 2},
        "expected": [float("nan"), float("nan"), 2 / 3, 2.0, 8 / 3, 14 / 3, 6.0],
    },
}


@pytest.mark.filterwarnings(
    "ignore:`Expr.rolling_var` is being called from the stable API although considered an unstable feature."
)
def test_rolling_var_expr(
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
            name: nw.col("a").rolling_var(**values["kwargs"])  # type: ignore[arg-type]
            for name, values in kwargs_and_expected.items()
        }
    )
    expected = {name: values["expected"] for name, values in kwargs_and_expected.items()}

    assert_equal_data(result, expected)


@pytest.mark.filterwarnings(
    "ignore:`Series.rolling_var` is being called from the stable API although considered an unstable feature."
)
def test_rolling_var_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result = df.select(
        **{
            name: df["a"].rolling_var(**values["kwargs"])  # type: ignore[arg-type]
            for name, values in kwargs_and_expected.items()
        }
    )
    expected = {name: values["expected"] for name, values in kwargs_and_expected.items()}
    assert_equal_data(result, expected)


@given(  # type: ignore[misc]
    center=st.booleans(),
    values=st.lists(st.floats(-10, 10), min_size=3, max_size=10),
    ddof=st.integers(min_value=0),
)
@pytest.mark.skipif(PANDAS_VERSION < (1,), reason="too old for pyarrow")
@pytest.mark.filterwarnings("ignore:.*:narwhals.exceptions.NarwhalsUnstableWarning")
def test_rolling_var_hypothesis(center: bool, values: list[float], ddof: int) -> None:  # noqa: FBT001
    s = pd.Series(values)
    n_missing = random.randint(0, len(s) - 1)  # noqa: S311
    window_size = random.randint(1, len(s))  # noqa: S311
    min_periods = random.randint(1, window_size)  # noqa: S311
    mask = random.sample(range(len(s)), n_missing)
    s[mask] = None
    df = pd.DataFrame({"a": s})
    expected = (
        s.rolling(window=window_size, center=center, min_periods=min_periods)
        .var(ddof=ddof)
        .to_frame("a")
    )
    result = nw.from_native(pa.Table.from_pandas(df)).select(
        nw.col("a").rolling_var(
            window_size, center=center, min_periods=min_periods, ddof=ddof
        )
    )
    expected_dict = nw.from_native(expected, eager_only=True).to_dict(as_series=False)
    assert_equal_data(result, expected_dict)
