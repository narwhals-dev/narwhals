from __future__ import annotations

import random
from typing import Any

import hypothesis.strategies as st
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from hypothesis import given

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION
from tests.utils import POLARS_VERSION
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0]}

kwargs_and_expected = (
    {
        "name": "x1",
        "kwargs": {"window_size": 3},
        "expected": [None, None, 1 / 3, 1, 4 / 3, 7 / 3, 3],
    },
    {
        "name": "x2",
        "kwargs": {"window_size": 3, "min_periods": 1},
        "expected": [None, 0.5, 1 / 3, 1.0, 4 / 3, 7 / 3, 3],
    },
    {
        "name": "x3",
        "kwargs": {"window_size": 2, "min_periods": 1},
        "expected": [None, 0.5, 0.5, 2.0, 2.0, 4.5, 4.5],
    },
    {
        "name": "x4",
        "kwargs": {"window_size": 5, "min_periods": 1, "center": True},
        "expected": [1 / 3, 11 / 12, 4 / 5, 17 / 10, 2.0, 2.25, 3],
    },
    {
        "name": "x5",
        "kwargs": {"window_size": 4, "min_periods": 1, "center": True},
        "expected": [0.5, 1 / 3, 11 / 12, 11 / 12, 2.25, 2.25, 3],
    },
    {
        "name": "x6",
        "kwargs": {"window_size": 3, "ddof": 2},
        "expected": [None, None, 2 / 3, 2.0, 8 / 3, 14 / 3, 6.0],
    },
)


@pytest.mark.filterwarnings(
    "ignore:`Expr.rolling_var` is being called from the stable API although considered an unstable feature."
)
@pytest.mark.parametrize("kwargs_and_expected", kwargs_and_expected)
def test_rolling_var_expr(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    kwargs_and_expected: dict[str, Any],
) -> None:
    name = kwargs_and_expected["name"]
    kwargs = kwargs_and_expected["kwargs"]
    expected = kwargs_and_expected["expected"]

    if "polars" in str(constructor_eager) and POLARS_VERSION < (1,):
        # TODO(FBruzzesi): Dask is raising the following error:
        # NotImplementedError: Partition size is less than overlapping window size.
        # Try using ``df.repartition`` to increase the partition size.
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").rolling_var(**kwargs).alias(name))

    assert_equal_data(result, {name: expected})


@pytest.mark.filterwarnings(
    "ignore:`Series.rolling_var` is being called from the stable API although considered an unstable feature."
)
@pytest.mark.parametrize("kwargs_and_expected", kwargs_and_expected)
def test_rolling_var_series(
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
    result = df.select(df["a"].rolling_var(**kwargs).alias(name))

    assert_equal_data(result, {name: expected})


@given(  # type: ignore[misc]
    center=st.booleans(),
    values=st.lists(st.floats(-10, 10), min_size=5, max_size=10),
)
@pytest.mark.skipif(PANDAS_VERSION < (1,), reason="too old for pyarrow")
@pytest.mark.skipif(POLARS_VERSION < (1,), reason="different null behavior")
@pytest.mark.filterwarnings("ignore:.*is_sparse is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*:narwhals.exceptions.NarwhalsUnstableWarning")
def test_rolling_var_hypothesis(center: bool, values: list[float]) -> None:  # noqa: FBT001
    s = pd.Series(values)
    window_size = random.randint(2, len(s))  # noqa: S311
    min_periods = random.randint(2, window_size)  # noqa: S311
    ddof = random.randint(0, min_periods - 1)  # noqa: S311
    mask = random.sample(range(len(s)), 2)

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

    result = nw.from_native(pl.from_pandas(df)).select(
        nw.col("a").rolling_var(
            window_size, center=center, min_periods=min_periods, ddof=ddof
        )
    )
    expected_dict = nw.from_native(expected, eager_only=True).to_dict(as_series=False)
    assert_equal_data(result, expected_dict)
