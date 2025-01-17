from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


@pytest.mark.parametrize("include_breakpoint", [True, False])
@pytest.mark.parametrize("include_category", [True, False])
@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist(
    constructor: ConstructorEager,
    *,
    include_breakpoint: bool,
    include_category: bool,
) -> None:
    data = {
        "int": [0, 1, 2, 3, 4, 5, 6],
        "float": [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
    }
    df = nw.from_native(constructor(data))

    bins = [-float("inf"), 2.5, 5.5, float("inf")]
    expected = {
        "breakpoint": [2.5, 5.5, float("inf")],
        "category": ["(-inf, 2.5]", "(2.5, 5.5]", "(5.5, inf]"],
        "count": [3, 3, 1],
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


@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_bins_and_bin_count() -> None:
    import polars as pl

    s = nw.from_native(pl.Series([1, 2, 3]), series_only=True)
    with pytest.raises(InvalidOperationError, match="must provide one of"):
        s.hist(bins=None, bin_count=None)

    with pytest.raises(InvalidOperationError, match="can only provide one of"):
        s.hist(bins=[1, 3], bin_count=4)


@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_non_monotonic(constructor: ConstructorEager) -> None:
    df = nw.from_native(constructor({"int": [0, 1, 2, 3, 4, 5, 6]}))

    with pytest.raises(Exception, match="monotonic"):
        df["int"].hist(bins=[5, 1, 0])
