from __future__ import annotations

from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import given

import narwhals.stable.v1 as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import POLARS_VERSION
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data
from tests.utils import nwise

data = {
    "int": [0, 1, 2, 3, 4, 5, 6],
    "float": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}

bins_and_expected = [
    {
        "bins": [-float("inf"), 2.5, 5.5, float("inf")],
        "expected": [3, 3, 1],
    },
    {
        "bins": [1.0, 2.5, 5.5, float("inf")],
        "expected": [1, 3, 1],
    },
    {
        "bins": [1.0, 2.5, 5.5],
        "expected": [1, 3],
    },
    {
        "bins": [-10.0, -1.0, 2.5, 5.5],
        "expected": [0, 3, 3],
    },
]
counts_and_expected = [
    {
        "bin_count": 4,
        "expected_bins": [-0.006, 1.5, 3.0, 4.5, 6.0],
        "expected_count": [2, 2, 1, 2],
    },
    {
        "bin_count": 12,
        "expected_bins": [
            -0.006,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
        ],
        "expected_count": [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    },
    {
        "bin_count": 0,
        "expected_bins": [],
        "expected_count": [],
    },
]


@pytest.mark.parametrize("params", bins_and_expected)
@pytest.mark.parametrize("include_breakpoint", [True, False])
@pytest.mark.parametrize("include_category", [True, False])
@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_bin(
    constructor_eager: ConstructorEager,
    *,
    params: dict[str, Any],
    include_breakpoint: bool,
    include_category: bool,
) -> None:
    df = nw.from_native(constructor_eager(data))
    bins = params["bins"]

    expected = {
        "breakpoint": bins[1:],
        "category": [f"({left}, {right}]" for left, right in nwise(bins, n=2)],
        "count": params["expected"],
    }
    if not include_breakpoint:
        del expected["breakpoint"]
    if not include_category:
        del expected["category"]

    result = df["int"].hist(
        bins=bins,
        include_breakpoint=include_breakpoint,
        include_category=include_category,
    )
    assert_equal_data(result, expected)

    result = df["float"].hist(
        bins=bins,
        include_breakpoint=include_breakpoint,
        include_category=include_category,
    )
    assert_equal_data(result, expected)


@pytest.mark.skipif(
    POLARS_VERSION < (1, 0),
    reason="hist(bin_count=...) behavior significantly changed after 1.0",
)
@pytest.mark.parametrize("params", counts_and_expected)
@pytest.mark.parametrize("include_breakpoint", [True, False])
@pytest.mark.parametrize("include_category", [True, False])
@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_count(
    constructor_eager: ConstructorEager,
    *,
    params: dict[str, Any],
    include_breakpoint: bool,
    include_category: bool,
) -> None:
    df = nw.from_native(constructor_eager(data))

    bins = params["expected_bins"]
    expected = {
        "breakpoint": bins[1:],
        "category": [f"({left}, {right}]" for left, right in nwise(bins, n=2)],
        "count": params["expected_count"],
    }
    if not include_breakpoint:
        del expected["breakpoint"]
    if not include_category:
        del expected["category"]

    result = df["int"].hist(
        bin_count=params["bin_count"],
        include_breakpoint=include_breakpoint,
        include_category=include_category,
    )
    assert_equal_data(result, expected)

    result = df["float"].hist(
        bin_count=params["bin_count"],
        include_breakpoint=include_breakpoint,
        include_category=include_category,
    )
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_bin_and_bin_count() -> None:
    import polars as pl

    s = nw.from_native(pl.Series([1, 2, 3]), series_only=True)
    with pytest.raises(InvalidOperationError, match="must provide one of"):
        s.hist(bins=None, bin_count=None)

    with pytest.raises(InvalidOperationError, match="can only provide one of"):
        s.hist(bins=[1, 3], bin_count=4)


@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_non_monotonic(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"int": [0, 1, 2, 3, 4, 5, 6]}))

    with pytest.raises(Exception, match="monotonic"):
        df["int"].hist(bins=[5, 0, 2])

    with pytest.raises(Exception, match="monotonic"):
        df["int"].hist(bins=[5, 2, 0])


@given(  # type: ignore[misc]
    data=st.lists(st.floats(min_value=-1_000, max_value=1_000), min_size=1, max_size=100),
    bin_deltas=st.lists(
        st.floats(min_value=0.001, max_value=1_000, allow_nan=False), max_size=50
    ),
)
@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
@pytest.mark.slow
def test_hist_bins_hypotheis(
    constructor_eager: ConstructorEager,
    data: list[float],
    bin_deltas: list[float],
) -> None:
    import polars as pl

    df = nw.from_native(constructor_eager({"values": data})).select(
        nw.col("values").cast(nw.Float64)
    )
    bins = (
        nw.from_native(constructor_eager({"bins": bin_deltas})["bins"], series_only=True)  # type:ignore[index]
        .cast(nw.Float64)
        .cum_sum()
    )

    result = df["values"].hist(
        bins=bins.to_list(),
        include_breakpoint=False,
        include_category=False,
    )
    expected = (
        pl.Series(data, dtype=pl.Float64)
        .hist(
            bins=pl.Series(bin_deltas, dtype=pl.Float64).cum_sum().to_list(),
            include_breakpoint=False,
            include_category=False,
        )
        .rename({"": "count"})
    ).to_dict(as_series=False)

    assert_equal_data(result, expected)


@given(  # type: ignore[misc]
    data=st.lists(
        st.floats(min_value=-1_000, max_value=1_000, allow_subnormal=False),
        min_size=1,
        max_size=100,
    ),
    bin_count=st.integers(min_value=0, max_value=1_000),
)
@pytest.mark.skipif(
    POLARS_VERSION < (1, 0),
    reason="hist(bin_count=...) behavior significantly changed after 1.0",
)
@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
@pytest.mark.slow
def test_hist_count_hypothesis(
    constructor_eager: ConstructorEager,
    data: list[float],
    bin_count: int,
    request: pytest.FixtureRequest,
) -> None:
    import polars as pl

    df = nw.from_native(constructor_eager({"values": data})).select(
        nw.col("values").cast(nw.Float64)
    )

    result = df["values"].hist(
        bin_count=bin_count,
        include_breakpoint=False,
        include_category=False,
    )
    expected = (
        pl.Series(data, dtype=pl.Float64)
        .hist(
            bin_count=bin_count,
            include_breakpoint=False,
            include_category=False,
        )
        .rename({"": "count"})
    )

    # Bug in Polars <= 1.2.0; hist becomes unreliable when passing bin_counts
    #   for data with a wide range and a large number of passed bins
    #   https://github.com/pola-rs/polars/issues/20879
    if expected["count"].sum() != len(data) and "polars" not in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    assert_equal_data(result, expected.to_dict(as_series=False))
