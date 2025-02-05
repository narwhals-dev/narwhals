from __future__ import annotations

from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import given

import narwhals.stable.v1 as nw
from narwhals.exceptions import ComputeError
from tests.utils import POLARS_VERSION
from tests.utils import PYARROW_VERSION
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {
    "int": [0, 1, 2, 3, 4, 5, 6],
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
    {
        "bins": [1.0, 2.0625],
        "expected": [1],
    },
    {
        "bins": [1],
        "expected": [],
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
@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_bin(
    constructor_eager: ConstructorEager,
    *,
    params: dict[str, Any],
    include_breakpoint: bool,
    request: pytest.FixtureRequest,
) -> None:
    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (13,):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data)).with_columns(
        float=nw.col("int").cast(nw.Float64),
    )
    bins = params["bins"]

    expected = {
        "breakpoint": bins[1:],
        "count": params["expected"],
    }
    if not include_breakpoint:
        del expected["breakpoint"]

    # smoke tests
    for col in df.columns:
        result = df[col].hist(
            bins=bins,
            include_breakpoint=include_breakpoint,
        )
        assert_equal_data(result, expected)

        # result size property
        if len(bins) < 2:
            assert len(result) == 0

    # shift bins property
    shift_by = 10
    shifted_bins = [b + shift_by for b in bins]
    expected = {
        "breakpoint": shifted_bins[1:],
        "count": params["expected"],
    }
    if not include_breakpoint:
        del expected["breakpoint"]

    for col in df.columns:
        result = (df[col] + shift_by).hist(
            bins=shifted_bins,
            include_breakpoint=include_breakpoint,
        )
        assert_equal_data(result, expected)

    # missing/nan results
    df = nw.from_native(
        constructor_eager(
            {
                "has_nan": [float("nan"), *data["int"]],
                "has_null": [None, *data["int"]],
            }
        )
    )
    bins = params["bins"]
    expected = {
        "breakpoint": bins[1:],
        "count": params["expected"],
    }
    if not include_breakpoint:
        del expected["breakpoint"]

    for col in df.columns:
        result = df[col].hist(
            bins=bins,
            include_breakpoint=include_breakpoint,
        )

        assert_equal_data(result, expected)


@pytest.mark.parametrize("params", counts_and_expected)
@pytest.mark.parametrize("include_breakpoint", [True, False])
@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_count(
    constructor_eager: ConstructorEager,
    *,
    params: dict[str, Any],
    include_breakpoint: bool,
    request: pytest.FixtureRequest,
) -> None:
    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (13,):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data)).with_columns(
        float=nw.col("int").cast(nw.Float64),
    )
    bin_count = params["bin_count"]

    expected_bins = params["expected_bins"]
    expected = {
        "breakpoint": expected_bins[1:],
        "count": params["expected_count"],
    }
    if not include_breakpoint:
        del expected["breakpoint"]

    # smoke tests
    for col in df.columns:
        result = df[col].hist(
            bin_count=bin_count,
            include_breakpoint=include_breakpoint,
        )
        assert_equal_data(result, expected)

        # result size property
        if bin_count < 2:
            assert len(result) == 0
        else:
            assert result["count"].sum() == df[col].count()

    # missing/nan results
    df = nw.from_native(
        constructor_eager(
            {
                "has_nan": [float("nan"), *data["int"]],
                "has_null": [None, *data["int"]],
            }
        )
    )

    for col in df.columns:
        result = df[col].hist(
            bin_count=bin_count,
            include_breakpoint=include_breakpoint,
        )
        assert_equal_data(result, expected)

        # result size property
        if bin_count < 2:
            assert len(result) == 0
        else:
            assert (
                result["count"].sum() == (~(df[col].is_nan() | df[col].is_null())).sum()
            )


@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_count_no_spread(
    constructor_eager: ConstructorEager,
) -> None:
    data = {
        "all_zero": [0, 0, 0],
        "all_non_zero": [5, 5, 5],
    }
    df = nw.from_native(constructor_eager(data))

    result = df["all_zero"].hist(bin_count=4, include_breakpoint=True)
    expected = {"breakpoint": [-0.25, 0.0, 0.25, 0.5], "count": [0, 3, 0, 0]}
    assert_equal_data(result, expected)

    result = df["all_non_zero"].hist(bin_count=4, include_breakpoint=True)
    expected = {"breakpoint": [4.75, 5.0, 5.25, 5.5], "count": [0, 3, 0, 0]}
    assert_equal_data(result, expected)

    result = df["all_zero"].hist(bin_count=1, include_breakpoint=True)
    expected = {
        "breakpoint": [0.5],
        "count": [3],
    }
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_bin_and_bin_count() -> None:
    import polars as pl

    s = nw.from_native(pl.Series([1, 2, 3]), series_only=True)
    result = s.hist(bins=None, bin_count=None)
    assert len(result) == 10

    with pytest.raises(ComputeError, match="can only provide one of"):
        s.hist(bins=[1, 3], bin_count=4)


@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_no_data(constructor_eager: ConstructorEager) -> None:
    s = nw.from_native(constructor_eager({"values": []})).select(
        nw.col("values").cast(nw.Float64)
    )["values"]
    for include_breakpoint in [True, False]:
        result = s.hist(bin_count=10, include_breakpoint=include_breakpoint)
        assert len(result) == 10
        assert result["count"].sum() == 0

    for include_breakpoint in [True, False]:
        result = s.hist(bins=[1, 5, 10], include_breakpoint=include_breakpoint)
        assert len(result) == 2
        assert result["count"].sum() == 0


@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_small_bins(constructor_eager: ConstructorEager) -> None:
    s = nw.from_native(constructor_eager({"values": [1, 2, 3]}))
    result = s["values"].hist(bins=None, bin_count=None)
    assert len(result) == 10

    with pytest.raises(ComputeError, match="can only provide one of"):
        s["values"].hist(bins=[1, 3], bin_count=4)


@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_non_monotonic(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"int": [0, 1, 2, 3, 4, 5, 6]}))

    with pytest.raises(ComputeError, match="monotonic"):
        df["int"].hist(bins=[5, 0, 2])

    with pytest.raises(ComputeError, match="monotonic"):
        df["int"].hist(bins=[5, 2, 0])


@given(  # type: ignore[misc]
    data=st.lists(
        # Bug in Polars <= 1.21; computing histograms with NaN data and passed bins can be unreliable
        #   this leads to flaky hypothesis testing https://github.com/pola-rs/polars/issues/21082
        # min_value/max_value from PyArrows min/max boundaries
        st.floats(
            min_value=-9007199254740992,
            max_value=9007199254740992,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=0,
        max_size=100,
    ),
    bin_deltas=st.lists(
        st.floats(min_value=0.001, max_value=1_000, allow_nan=False), max_size=50
    ),
)
@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature.",
    "ignore:invalid value encountered in cast:RuntimeWarning",
)
@pytest.mark.skipif(
    POLARS_VERSION < (1, 15),
    reason="hist(bins=...) cannot be used for compatibility checks since narwhals aims to mimic polars>=1.2.0 behavior",
)
@pytest.mark.slow
def test_hist_bin_hypotheis(
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
        include_breakpoint=True,
    )
    expected_data = pl.Series(data, dtype=pl.Float64)
    expected = expected_data.hist(
        bins=bins.to_list(),
        include_breakpoint=True,
        include_category=False,
    )

    assert_equal_data(result, expected.to_dict(as_series=False))


@given(  # type: ignore[misc]
    data=st.lists(
        # min_value/max_value from PyArrows min/max boundaries
        st.floats(
            min_value=-9007199254740992,
            max_value=9007199254740992,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=0,
        max_size=100,
    ),
    bin_count=st.integers(min_value=0, max_value=1_000),
)
@pytest.mark.skipif(
    POLARS_VERSION < (1, 15),
    reason="hist(bin_count=...) behavior significantly changed after this version",
)
@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature.",
    "ignore:invalid value encountered in cast:RuntimeWarning",
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

    try:
        result = df["values"].hist(
            bin_count=bin_count,
            include_breakpoint=True,
        )
    except pl.exceptions.PanicException:  # pragma: no cover
        # panic occurs from specific float inputs on Polars 1.15
        if (1, 14) < POLARS_VERSION < (1, 16):
            request.applymarker(pytest.mark.xfail)
        raise

    expected_data = pl.Series(data, dtype=pl.Float64)
    expected = expected_data.hist(
        bin_count=bin_count,
        include_breakpoint=True,
        include_category=False,
    )

    # Bug in Polars <= 1.21; hist becomes unreliable when passing bin_counts
    #   for data with a wide range and a large number of passed bins
    #   https://github.com/pola-rs/polars/issues/20879
    if expected["count"].sum() != expected_data.count() and "polars" not in str(
        constructor_eager
    ):
        request.applymarker(pytest.mark.xfail)

    assert_equal_data(result, expected.to_dict(as_series=False))
