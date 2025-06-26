from __future__ import annotations

import random
from typing import Any

import hypothesis.strategies as st
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import (
    DUCKDB_VERSION,
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {"a": [None, 1, 2, None, 4, 6, 11]}

kwargs_and_expected: dict[str, dict[str, Any]] = {
    "x1": {"kwargs": {"window_size": 3}, "expected": [None] * 6 + [21]},
    "x2": {
        "kwargs": {"window_size": 3, "min_samples": 1},
        "expected": [None, 1.0, 3.0, 3.0, 6.0, 10.0, 21.0],
    },
    "x3": {
        "kwargs": {"window_size": 2, "min_samples": 1},
        "expected": [None, 1.0, 3.0, 2.0, 4.0, 10.0, 17.0],
    },
    "x4": {
        "kwargs": {"window_size": 5, "min_samples": 1, "center": True},
        "expected": [3.0, 3.0, 7.0, 13.0, 23.0, 21.0, 21.0],
    },
    "x5": {
        "kwargs": {"window_size": 4, "min_samples": 1, "center": True},
        "expected": [1.0, 3.0, 3.0, 7.0, 12.0, 21.0, 21.0],
    },
}


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


@pytest.mark.parametrize(
    ("expected_a", "window_size", "min_samples", "center"),
    [
        ([None, None, 3, None, None, 10, 17], 2, None, False),
        ([None, None, 3, None, None, 10, 17], 2, 2, False),
        ([None, None, 3, 3, 6, 10, 21], 3, 2, False),
        ([1, None, 3, 3, 6, 10, 21], 3, 1, False),
        ([3, 1, 3, 6, 10, 21, 17], 3, 1, True),
        ([3, 1, 3, 7, 12, 21, 21], 4, 1, True),
        ([3, 3, 7, 13, 23, 21, 21], 5, 1, True),
    ],
)
def test_rolling_sum_expr_lazy_ungrouped(
    constructor: Constructor,
    expected_a: list[float],
    window_size: int,
    min_samples: int,
    *,
    center: bool,
) -> None:
    if ("polars" in str(constructor) and POLARS_VERSION < (1, 10)) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3)
    ):
        pytest.skip()
    if "modin" in str(constructor):
        # unreliable
        pytest.skip()
    data = {
        "a": [1, None, 2, None, 4, 6, 11],
        "b": [1, None, 2, 3, 4, 5, 6],
        "i": list(range(7)),
    }
    df = nw.from_native(constructor(data))
    result = (
        df.with_columns(
            nw.col("a")
            .rolling_sum(window_size, min_samples=min_samples, center=center)
            .over(order_by="b")
        )
        .select("a", "i")
        .sort("i")
    )
    expected = {"a": expected_a, "i": list(range(7))}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("expected_a", "window_size", "min_samples", "center"),
    [
        ([None, None, 3, None, None, 10, 17], 2, None, False),
        ([None, None, 3, None, None, 10, 17], 2, 2, False),
        ([None, None, 3, 3, None, 10, 21], 3, 2, False),
        ([1, None, 3, 3, 4, 10, 21], 3, 1, False),
        ([3, 1, 3, 2, 10, 21, 17], 3, 1, True),
        ([3, 1, 3, 3, 10, 21, 21], 4, 1, True),
        ([3, 3, 3, 3, 21, 21, 21], 5, 1, True),
    ],
)
def test_rolling_sum_expr_lazy_grouped(
    constructor: Constructor,
    expected_a: list[float],
    window_size: int,
    min_samples: int,
    request: pytest.FixtureRequest,
    *,
    center: bool,
) -> None:
    if ("polars" in str(constructor) and POLARS_VERSION < (1, 10)) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3)
    ):
        pytest.skip()
    if "pandas" in str(constructor) and PANDAS_VERSION < (1, 2):
        pytest.skip()
    if any(x in str(constructor) for x in ("dask", "pyarrow_table")):
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(constructor):
        # unreliable
        pytest.skip()
    data = {
        "a": [1, None, 2, None, 4, 6, 11],
        "g": [1, 1, 1, 1, 2, 2, 2],
        "b": [1, None, 2, 3, 4, 5, 6],
        "i": list(range(7)),
    }
    df = nw.from_native(constructor(data))
    result = (
        df.with_columns(
            nw.col("a")
            .rolling_sum(window_size, min_samples=min_samples, center=center)
            .over("g", order_by="b")
        )
        .sort("i")
        .select("a")
    )
    expected = {"a": expected_a}
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


@pytest.mark.parametrize(
    ("window_size", "min_samples", "context"),
    [
        (
            -1,
            None,
            pytest.raises(
                ValueError, match="window_size must be greater or equal than 1"
            ),
        ),
        (
            2,
            -1,
            pytest.raises(
                ValueError, match="min_samples must be greater or equal than 1"
            ),
        ),
        (
            1,
            2,
            pytest.raises(
                InvalidOperationError,
                match="`min_samples` must be less or equal than `window_size`",
            ),
        ),
        (
            4.2,
            None,
            pytest.raises(TypeError, match=r"Expected '.+?', got: '.+?'\s+window_size="),
        ),
        (
            2,
            4.2,
            pytest.raises(TypeError, match=r"Expected '.+?', got: '.+?'\s+min_samples="),
        ),
    ],
)
def test_rolling_sum_expr_invalid_params(
    constructor_eager: ConstructorEager,
    window_size: int,
    min_samples: int | None,
    context: Any,
) -> None:
    df = nw.from_native(constructor_eager(data))

    with context:
        df.select(
            nw.col("a").rolling_sum(window_size=window_size, min_samples=min_samples)
        )


@pytest.mark.filterwarnings(
    "ignore:`Series.rolling_sum` is being called from the stable API although considered an unstable feature."
)
@pytest.mark.parametrize(
    ("window_size", "min_samples", "context"),
    [
        (
            -1,
            None,
            pytest.raises(
                ValueError, match="window_size must be greater or equal than 1"
            ),
        ),
        (
            2,
            -1,
            pytest.raises(
                ValueError, match="min_samples must be greater or equal than 1"
            ),
        ),
        (
            1,
            2,
            pytest.raises(
                InvalidOperationError,
                match="`min_samples` must be less or equal than `window_size`",
            ),
        ),
        (
            4.2,
            None,
            pytest.raises(TypeError, match=r"Expected '.+?', got: '.+?'\s+window_size="),
        ),
        (
            2,
            4.2,
            pytest.raises(TypeError, match=r"Expected '.+?', got: '.+?'\s+min_samples="),
        ),
    ],
)
def test_rolling_sum_series_invalid_params(
    constructor_eager: ConstructorEager,
    window_size: int,
    min_samples: int | None,
    context: Any,
) -> None:
    df = nw.from_native(constructor_eager(data))

    with context:
        df["a"].rolling_sum(window_size=window_size, min_samples=min_samples)


@given(center=st.booleans(), values=st.lists(st.floats(-10, 10), min_size=3, max_size=10))
@pytest.mark.filterwarnings("ignore:.*:narwhals.exceptions.NarwhalsUnstableWarning")
@pytest.mark.filterwarnings("ignore:.*is_sparse is deprecated:DeprecationWarning")
@pytest.mark.slow
def test_rolling_sum_hypothesis(center: bool, values: list[float]) -> None:  # noqa: FBT001
    s = pd.Series(values)
    n_missing = random.randint(0, len(s) - 1)  # noqa: S311
    window_size = random.randint(1, len(s))  # noqa: S311
    min_samples = random.randint(1, window_size)  # noqa: S311
    mask = random.sample(range(len(s)), n_missing)
    s[mask] = None
    df = pd.DataFrame({"a": s})
    expected = (
        s.rolling(window=window_size, center=center, min_periods=min_samples)
        .sum()
        .to_frame("a")
    )
    result = nw.from_native(pa.Table.from_pandas(df)).select(
        nw.col("a").rolling_sum(window_size, center=center, min_samples=min_samples)
    )
    expected_dict = nw.from_native(expected, eager_only=True).to_dict(as_series=False)
    assert_equal_data(result, expected_dict)
