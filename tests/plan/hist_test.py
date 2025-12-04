# TODO(unassigned): cudf has too many spurious failures. Report and revisit?
# Modin is too slow so is excluded.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals.typing import EagerAllowed
    from tests.conftest import Data


XFAIL_HIST_NOT_IMPLEMENTED = pytest.mark.xfail(
    reason="`ArrowExpr.hist_*` is not yet implemented"
)


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "int": [0, 1, 2, 3, 4, 5, 6],
        "float": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "int_shuffled": [1, 0, 2, 3, 6, 5, 4],
        "float_shuffled": [1.0, 0.0, 2.0, 3.0, 6.0, 5.0, 4.0],
    }


@pytest.fixture(scope="module")
def data_missing(data: Data) -> Data:
    return {"has_nan": [float("nan"), *data["int"]], "has_null": [None, *data["int"]]}


@pytest.fixture(scope="module", params=["pyarrow"])
def backend(request: pytest.FixtureRequest) -> EagerAllowed:
    result: EagerAllowed = request.param
    return result


@pytest.fixture(
    scope="module", params=[True, False], ids=["breakpoint-True", "breakpoint-False"]
)
def include_breakpoint(request: pytest.FixtureRequest) -> bool:
    result: bool = request.param
    return result


counts_and_expected = [
    {
        "bin_count": 4,
        "expected_bins": [0, 1.5, 3.0, 4.5, 6.0],
        "expected_count": [2, 2, 1, 2],
    },
    {
        "bin_count": 12,
        "expected_bins": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        "expected_count": [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    },
    {"bin_count": 1, "expected_bins": [0, 6], "expected_count": [7]},
    {"bin_count": 0, "expected_bins": [], "expected_count": []},
]


SHIFT_BINS_BY = 10
"""shift bins property"""


# TODO @dangotbanned: Try to avoid all this looping (3x `iter_columns` in a single test?)
@XFAIL_HIST_NOT_IMPLEMENTED
@pytest.mark.parametrize(
    ("bins", "expected"),
    [
        ([-float("inf"), 2.5, 5.5, float("inf")], [3, 3, 1]),
        ([1.0, 2.5, 5.5, float("inf")], [2, 3, 1]),
        ([1.0, 2.5, 5.5], [2, 3]),
        ([-10.0, -1.0, 2.5, 5.5], [0, 3, 3]),
        ([1.0, 2.0625], [2]),
        ([1], []),
        ([0, 10], [7]),
    ],
    ids=str,
)
def test_hist_bin(
    data: Data,
    data_missing: Data,
    backend: EagerAllowed,
    bins: list[float],
    expected: Sequence[float],
    *,
    include_breakpoint: bool,
) -> None:
    df = nwp.DataFrame.from_dict(data, backend=backend).with_columns(
        float=nwp.col("int").cast(nw.Float64)
    )
    expected_full = {"count": expected}
    if include_breakpoint:
        expected_full = {"breakpoint": bins[1:], **expected_full}
    # smoke tests
    for series in df.iter_columns():
        result = series.hist(bins=bins, include_breakpoint=include_breakpoint)
        assert_equal_data(result, expected_full)

        # result size property
        assert len(result) == max(len(bins) - 1, 0)

    # shift bins property
    shifted_bins = [b + SHIFT_BINS_BY for b in bins]
    expected_full = {"count": expected}
    if include_breakpoint:
        expected_full = {"breakpoint": shifted_bins[1:], **expected_full}

    for series in df.iter_columns():
        result = (series + SHIFT_BINS_BY).hist(
            shifted_bins, include_breakpoint=include_breakpoint
        )
        assert_equal_data(result, expected_full)

    # missing/nan results
    df = nwp.DataFrame.from_dict(data_missing, backend=backend)
    expected_full = {"count": expected}
    if include_breakpoint:
        expected_full = {"breakpoint": bins[1:], **expected_full}
    for series in df.iter_columns():
        result = series.hist(bins, include_breakpoint=include_breakpoint)
        assert_equal_data(result, expected_full)


# TODO @dangotbanned: Avoid using `del`
# TODO @dangotbanned: Split up `params`
# TODO @dangotbanned: Try to avoid all this looping
@XFAIL_HIST_NOT_IMPLEMENTED
@pytest.mark.parametrize("params", counts_and_expected)
def test_hist_count(
    data: Data,
    data_missing: Data,
    backend: EagerAllowed,
    *,
    params: dict[str, Any],
    include_breakpoint: bool,
) -> None:
    df = nwp.DataFrame.from_dict(data, backend=backend).with_columns(
        float=nwp.col("int").cast(nw.Float64)
    )
    bin_count = params["bin_count"]

    expected_bins = params["expected_bins"]
    expected = {"breakpoint": expected_bins[1:], "count": params["expected_count"]}
    if not include_breakpoint:
        del expected["breakpoint"]

    # smoke tests
    for col in df.columns:
        result = df.get_column(col).hist(
            bin_count=bin_count, include_breakpoint=include_breakpoint
        )
        assert_equal_data(result, expected)

        # result size property

        assert len(result) == bin_count
        if bin_count > 0:
            assert result.get_column("count").sum() == df.get_column(col).count()

    # missing/nan results
    df = nwp.DataFrame.from_dict(data_missing, backend=backend)

    for col in df.columns:
        result = df.get_column(col).hist(
            bin_count=bin_count, include_breakpoint=include_breakpoint
        )
        assert_equal_data(result, expected)

        # result size property
        assert len(result) == bin_count
        ser = df.get_column(col)
        if bin_count > 0:
            # NOTE: Could this just be a filter?
            assert (
                result.get_column("count").sum()
                == (~(ser.is_nan() | ser.is_null())).sum()
            )


# TODO @dangotbanned: parametrize into 3 cases
@XFAIL_HIST_NOT_IMPLEMENTED
def test_hist_count_no_spread(backend: EagerAllowed) -> None:
    data_ = {"all_zero": [0, 0, 0], "all_non_zero": [5, 5, 5]}
    df = nwp.DataFrame.from_dict(data_, backend=backend)

    result = df.get_column("all_zero").hist(bin_count=4, include_breakpoint=True)
    expected = {"breakpoint": [-0.25, 0.0, 0.25, 0.5], "count": [0, 3, 0, 0]}
    assert_equal_data(result, expected)

    result = df.get_column("all_non_zero").hist(bin_count=4, include_breakpoint=True)
    expected = {"breakpoint": [4.75, 5.0, 5.25, 5.5], "count": [0, 3, 0, 0]}
    assert_equal_data(result, expected)

    result = df.get_column("all_zero").hist(bin_count=1, include_breakpoint=True)
    expected = {"breakpoint": [0.5], "count": [3]}
    assert_equal_data(result, expected)


# TODO @dangotbanned: parametrize into 2 cases?
@XFAIL_HIST_NOT_IMPLEMENTED
def test_hist_no_data(backend: EagerAllowed, *, include_breakpoint: bool) -> None:
    data_: Data = {"values": []}
    df = nwp.DataFrame.from_dict(data_, {"values": nw.Float64()}, backend=backend)
    s = df.to_series()
    for bin_count in [1, 10]:
        result = s.hist(bin_count=bin_count, include_breakpoint=include_breakpoint)
        assert len(result) == bin_count
        assert result.get_column("count").sum() == 0

        if include_breakpoint:
            bps = result.get_column("breakpoint").to_list()
            assert bps[0] == (1 / bin_count)
            if bin_count > 1:
                assert bps[-1] == 1

    result = s.hist(bins=[1, 5, 10], include_breakpoint=include_breakpoint)
    assert len(result) == 2
    assert result.get_column("count").sum() == 0


@XFAIL_HIST_NOT_IMPLEMENTED
def test_hist_small_bins(backend: EagerAllowed) -> None:
    s = nwp.Series.from_iterable([1, 2, 3], name="values", backend=backend)
    result = s.hist(bins=None, bin_count=None)
    assert len(result) == 10
