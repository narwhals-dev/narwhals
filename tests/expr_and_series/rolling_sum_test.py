from __future__ import annotations

import random
from typing import Any

import hypothesis.strategies as st
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given

import narwhals.stable.v1 as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import PANDAS_VERSION
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [None, 1, 2, None, 4, 6, 11]}

kwargs_and_expected: dict[str, dict[str, Any]] = {
    "x1": {"kwargs": {"window_size": 3}, "expected": [None] * 6 + [21]},
    "x2": {
        "kwargs": {"window_size": 3, "min_periods": 1},
        "expected": [None, 1.0, 3.0, 3.0, 6.0, 10.0, 21.0],
    },
    "x3": {
        "kwargs": {"window_size": 2, "min_periods": 1},
        "expected": [None, 1.0, 3.0, 2.0, 4.0, 10.0, 17.0],
    },
    "x4": {
        "kwargs": {"window_size": 5, "min_periods": 1, "center": True},
        "expected": [3.0, 3.0, 7.0, 13.0, 23.0, 21.0, 21.0],
    },
    "x5": {
        "kwargs": {"window_size": 4, "min_periods": 1, "center": True},
        "expected": [1.0, 3.0, 3.0, 7.0, 12.0, 21.0, 21.0],
    },
}


@pytest.mark.filterwarnings(
    "ignore:`Expr.rolling_sum` is being called from the stable API although considered an unstable feature."
)
def test_rolling_sum_expr(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.select(
        **{
            name: nw.col("a").rolling_sum(**values["kwargs"])
            for name, values in kwargs_and_expected.items()
        }
    )
    expected = {name: values["expected"] for name, values in kwargs_and_expected.items()}

    assert_equal_data(result, expected)


@pytest.mark.filterwarnings(
    "ignore:`Series.rolling_sum` is being called from the stable API although considered an unstable feature."
)
def test_rolling_sum_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result = df.select(
        **{
            name: df["a"].rolling_sum(**values["kwargs"])
            for name, values in kwargs_and_expected.items()
        }
    )
    expected = {name: values["expected"] for name, values in kwargs_and_expected.items()}
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings(
    "ignore:`Expr.rolling_sum` is being called from the stable API although considered an unstable feature."
)
@pytest.mark.parametrize(
    ("window_size", "min_periods", "context"),
    [
        (
            -1,
            None,
            pytest.raises(
                ValueError, match="window_size must be greater or equal than 1"
            ),
        ),
        (
            4.2,
            None,
            pytest.raises(
                TypeError,
                match="argument 'window_size': 'float' object cannot be interpreted as an integer",
            ),
        ),
        (
            2,
            -1,
            pytest.raises(
                ValueError, match="min_periods must be greater or equal than 1"
            ),
        ),
        (
            2,
            4.2,
            pytest.raises(
                TypeError,
                match="argument 'min_periods': 'float' object cannot be interpreted as an integer",
            ),
        ),
        (
            1,
            2,
            pytest.raises(
                InvalidOperationError,
                match="`min_periods` must be less or equal than `window_size`",
            ),
        ),
    ],
)
def test_rolling_sum_expr_invalid_params(
    constructor_eager: ConstructorEager,
    window_size: int,
    min_periods: int | None,
    context: Any,
) -> None:
    df = nw.from_native(constructor_eager(data))

    with context:
        df.select(
            nw.col("a").rolling_sum(window_size=window_size, min_periods=min_periods)
        )


@pytest.mark.filterwarnings(
    "ignore:`Series.rolling_sum` is being called from the stable API although considered an unstable feature."
)
@pytest.mark.parametrize(
    ("window_size", "min_periods", "context"),
    [
        (
            -1,
            None,
            pytest.raises(
                ValueError, match="window_size must be greater or equal than 1"
            ),
        ),
        (
            4.2,
            None,
            pytest.raises(
                TypeError,
                match="argument 'window_size': 'float' object cannot be interpreted as an integer",
            ),
        ),
        (
            2,
            -1,
            pytest.raises(
                ValueError, match="min_periods must be greater or equal than 1"
            ),
        ),
        (
            2,
            4.2,
            pytest.raises(
                TypeError,
                match="argument 'min_periods': 'float' object cannot be interpreted as an integer",
            ),
        ),
        (
            1,
            2,
            pytest.raises(
                InvalidOperationError,
                match="`min_periods` must be less or equal than `window_size`",
            ),
        ),
    ],
)
def test_rolling_sum_series_invalid_params(
    constructor_eager: ConstructorEager,
    window_size: int,
    min_periods: int | None,
    context: Any,
) -> None:
    df = nw.from_native(constructor_eager(data))

    with context:
        df["a"].rolling_sum(window_size=window_size, min_periods=min_periods)


@given(  # type: ignore[misc]
    center=st.booleans(), values=st.lists(st.floats(-10, 10), min_size=3, max_size=10)
)
@pytest.mark.skipif(PANDAS_VERSION < (1,), reason="too old for pyarrow")
@pytest.mark.filterwarnings("ignore:.*:narwhals.exceptions.NarwhalsUnstableWarning")
@pytest.mark.filterwarnings("ignore:.*is_sparse is deprecated:DeprecationWarning")
@pytest.mark.slow
def test_rolling_sum_hypothesis(center: bool, values: list[float]) -> None:  # noqa: FBT001
    s = pd.Series(values)
    n_missing = random.randint(0, len(s) - 1)  # noqa: S311
    window_size = random.randint(1, len(s))  # noqa: S311
    min_periods = random.randint(1, window_size)  # noqa: S311
    mask = random.sample(range(len(s)), n_missing)
    s[mask] = None
    df = pd.DataFrame({"a": s})
    expected = (
        s.rolling(window=window_size, center=center, min_periods=min_periods)
        .sum()
        .to_frame("a")
    )
    result = nw.from_native(pa.Table.from_pandas(df)).select(
        nw.col("a").rolling_sum(window_size, center=center, min_periods=min_periods)
    )
    expected_dict = nw.from_native(expected, eager_only=True).to_dict(as_series=False)
    assert_equal_data(result, expected_dict)
