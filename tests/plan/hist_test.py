from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals.typing import EagerAllowed, IntoDType
    from tests.conftest import Data

pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "int": [0, 1, 2, 3, 4, 5, 6],
        "float": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "int_shuffled": [1, 0, 2, 3, 6, 5, 4],
        "float_shuffled": [1.0, 0.0, 2.0, 3.0, 6.0, 5.0, 4.0],
    }


@pytest.fixture(scope="module")
def schema_data() -> nw.Schema:
    return nw.Schema(
        {
            "int": nw.Int64(),
            "float": nw.Float64(),
            "int_shuffled": nw.Int64(),
            "float_shuffled": nw.Float64(),
        }
    )


@pytest.fixture(scope="module", params=["int", "float", "int_shuffled", "float_shuffled"])
def column_data(request: pytest.FixtureRequest) -> str:
    result: str = request.param
    return result


@pytest.fixture(scope="module")
def data_missing(data: Data) -> Data:
    return {"has_nan": [float("nan"), *data["int"]], "has_null": [None, *data["int"]]}


@pytest.fixture(scope="module")
def schema_data_missing() -> nw.Schema:
    return nw.Schema({"has_nan": nw.Float64(), "has_null": nw.Int64()})


@pytest.fixture(scope="module", params=["has_nan", "has_null"])
def column_data_missing(request: pytest.FixtureRequest) -> str:
    result: str = request.param
    return result


@pytest.fixture(scope="module", params=["pyarrow"])
def backend(request: pytest.FixtureRequest) -> EagerAllowed:
    result: EagerAllowed = request.param
    return result


@pytest.fixture(scope="module", params=[True, False])
def include_breakpoint(request: pytest.FixtureRequest) -> bool:
    result: bool = request.param
    return result


def _series(
    name: str, source: Data, schema: nw.Schema, backend: EagerAllowed, /
) -> nwp.Series[Any]:
    values, dtype = (source[name], schema[name])
    return nwp.Series.from_iterable(values, name=name, dtype=dtype, backend=backend)


def _expected(
    bins: Sequence[float], count: Sequence[int], *, include_breakpoint: bool
) -> dict[str, Any]:
    if not include_breakpoint:
        return {"count": count}
    return {"breakpoint": bins[1:] if len(bins) > len(count) else bins, "count": count}


SHIFT_BINS_BY = 10
"""shift bins property"""

bins_cases = pytest.mark.parametrize(
    ("bins", "expected_count"),
    [
        pytest.param(
            [-float("inf"), 2.5, 5.5, float("inf")], [3, 3, 1], id="4_bins-neg-inf-inf"
        ),
        pytest.param([1.0, 2.5, 5.5, float("inf")], [2, 3, 1], id="4_bins-inf"),
        pytest.param([1.0, 2.5, 5.5], [2, 3], id="3_bins"),
        pytest.param([-10.0, -1.0, 2.5, 5.5], [0, 3, 3], id="4_bins"),
        pytest.param([1.0, 2.0625], [2], id="2_bins-1"),
        pytest.param([1], [], id="1_bins"),
        pytest.param([0, 10], [7], id="2_bins-2"),
    ],
)


@bins_cases
def test_hist_bins(
    data: Data,
    schema_data: nw.Schema,
    backend: EagerAllowed,
    column_data: str,
    bins: Sequence[float],
    expected_count: Sequence[int],
    *,
    include_breakpoint: bool,
) -> None:
    ser = _series(column_data, data, schema_data, backend)
    expected = _expected(bins, expected_count, include_breakpoint=include_breakpoint)
    result = ser.hist(bins, include_breakpoint=include_breakpoint)
    assert_equal_data(result, expected)
    assert len(result) == max(len(bins) - 1, 0)


@bins_cases
def test_hist_bins_shifted(
    data: Data,
    schema_data: nw.Schema,
    backend: EagerAllowed,
    column_data: str,
    bins: Sequence[float],
    expected_count: Sequence[int],
    *,
    include_breakpoint: bool,
) -> None:
    shifted_bins = [b + SHIFT_BINS_BY for b in bins]
    expected = _expected(
        shifted_bins, expected_count, include_breakpoint=include_breakpoint
    )
    ser = _series(column_data, data, schema_data, backend) + SHIFT_BINS_BY
    result = ser.hist(shifted_bins, include_breakpoint=include_breakpoint)
    assert_equal_data(result, expected)


@bins_cases
def test_hist_bins_missing(
    data_missing: Data,
    schema_data_missing: nw.Schema,
    backend: EagerAllowed,
    column_data_missing: str,
    bins: Sequence[float],
    expected_count: Sequence[int],
    *,
    include_breakpoint: bool,
) -> None:
    ser = _series(column_data_missing, data_missing, schema_data_missing, backend)
    expected = _expected(bins, expected_count, include_breakpoint=include_breakpoint)
    result = ser.hist(bins, include_breakpoint=include_breakpoint)
    assert_equal_data(result, expected)


bin_count_cases = pytest.mark.parametrize(
    ("bin_count", "expected_bins", "expected_count"),
    [
        (4, [1.5, 3.0, 4.5, 6.0], [2, 2, 1, 2]),
        (
            12,
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ),
        (1, [6], [7]),
        (0, [], []),
    ],
)


@bin_count_cases
def test_hist_bin_count(
    data: Data,
    schema_data: nw.Schema,
    backend: EagerAllowed,
    column_data: str,
    bin_count: int,
    expected_bins: Sequence[float],
    expected_count: Sequence[int],
    *,
    include_breakpoint: bool,
) -> None:
    ser = _series(column_data, data, schema_data, backend)
    expected = _expected(
        expected_bins, expected_count, include_breakpoint=include_breakpoint
    )
    result = ser.hist(bin_count=bin_count, include_breakpoint=include_breakpoint)
    assert_equal_data(result, expected)
    assert len(result) == bin_count
    if bin_count > 0:
        assert result.get_column("count").sum() == ser.drop_nans().count()


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            nwp.all().hist(bin_count=5),
            {
                "int": [2, 1, 1, 1, 2],
                "float": [2, 1, 1, 1, 2],
                "int_shuffled": [2, 1, 1, 1, 2],
                "float_shuffled": [2, 1, 1, 1, 2],
            },
        ),
        (
            (99 + nwp.all()).hist(bin_count=2).name.keep(),
            {
                "int": [4, 3],
                "float": [4, 3],
                "int_shuffled": [4, 3],
                "float_shuffled": [4, 3],
            },
        ),
        (
            nwp.all().hist([-3, -2, 3, 6]).name.to_uppercase(),
            {
                "INT": [0, 4, 3],
                "FLOAT": [0, 4, 3],
                "INT_SHUFFLED": [0, 4, 3],
                "FLOAT_SHUFFLED": [0, 4, 3],
            },
        ),
        (
            nwp.all().clip(upper_bound=4).hist([2, 3, 4, 5, 6]),
            {
                "int": [2, 3, 0, 0],
                "float": [2, 3, 0, 0],
                "int_shuffled": [2, 3, 0, 0],
                "float_shuffled": [2, 3, 0, 0],
            },
        ),
        (
            (nwp.all() * 2.7).hist([1.3, 5.1, 8.98, 11.3]),
            {
                "int": [1, 2, 1],
                "float": [1, 2, 1],
                "int_shuffled": [1, 2, 1],
                "float_shuffled": [1, 2, 1],
            },
        ),
    ],
)
def test_hist_expr_counts_only(
    data: Data,
    schema_data: nw.Schema,
    backend: EagerAllowed,
    expr: nwp.Expr,
    expected: dict[str, Any],
) -> None:
    df = nwp.DataFrame.from_dict(data, schema_data, backend=backend)
    result = df.select(expr)
    assert_equal_data(result, expected)


def test_hist_expr_breakpoint(
    data: Data, schema_data: nw.Schema, backend: EagerAllowed
) -> None:
    df = nwp.DataFrame.from_dict(data, schema_data, backend=backend)
    expr = nwp.all().hist(bin_count=3, include_breakpoint=True)
    result = df.select(expr)
    result_schema = result.collect_schema()

    dtype_breakpoint: IntoDType = nw.Float64
    # NOTE: To match polars it would be this, but maybe i64 is okay?
    dtype_count: IntoDType = nw.UInt32
    dtype_count = nw.Int64

    dtype_struct = nw.Struct({"breakpoint": dtype_breakpoint, "count": dtype_count})
    schema_struct = nw.Schema({"breakpoint": nw.Float64(), "count": nw.Int64()})
    expected_schema = nw.Schema(
        [
            ("int", dtype_struct),
            ("float", dtype_struct),
            ("int_shuffled", dtype_struct),
            ("float_shuffled", dtype_struct),
        ]
    )
    expected_data = {
        "int": [
            {"breakpoint": 2.0, "count": 3},
            {"breakpoint": 4.0, "count": 2},
            {"breakpoint": 6.0, "count": 2},
        ],
        "float": [
            {"breakpoint": 2.0, "count": 3},
            {"breakpoint": 4.0, "count": 2},
            {"breakpoint": 6.0, "count": 2},
        ],
        "int_shuffled": [
            {"breakpoint": 2.0, "count": 3},
            {"breakpoint": 4.0, "count": 2},
            {"breakpoint": 6.0, "count": 2},
        ],
        "float_shuffled": [
            {"breakpoint": 2.0, "count": 3},
            {"breakpoint": 4.0, "count": 2},
            {"breakpoint": 6.0, "count": 2},
        ],
    }
    assert result_schema == expected_schema
    assert_equal_data(result, expected_data)
    for ser in result.iter_columns():
        assert ser.struct.schema == schema_struct


@bin_count_cases
def test_hist_bin_count_missing(
    data_missing: Data,
    schema_data_missing: nw.Schema,
    backend: EagerAllowed,
    column_data_missing: str,
    bin_count: int,
    expected_bins: Sequence[float],
    expected_count: Sequence[int],
    *,
    include_breakpoint: bool,
) -> None:
    ser = _series(column_data_missing, data_missing, schema_data_missing, backend)
    expected = _expected(
        expected_bins, expected_count, include_breakpoint=include_breakpoint
    )
    result = ser.hist(bin_count=bin_count, include_breakpoint=include_breakpoint)
    assert_equal_data(result, expected)
    assert len(result) == bin_count
    if bin_count > 0:
        assert result.get_column("count").sum() == ser.drop_nans().count()


@pytest.mark.parametrize(
    ("column", "bin_count", "expected_breakpoint", "expected_count"),
    [
        ("all_zero", 4, [-0.25, 0.0, 0.25, 0.5], [0, 3, 0, 0]),
        ("all_non_zero", 4, [4.75, 5.0, 5.25, 5.5], [0, 3, 0, 0]),
        ("all_zero", 1, [0.5], [3]),
    ],
)
def test_hist_bin_count_no_spread(
    backend: EagerAllowed,
    column: str,
    bin_count: int,
    expected_breakpoint: Sequence[float],
    expected_count: Sequence[int],
) -> None:
    data = {"all_zero": [0, 0, 0], "all_non_zero": [5, 5, 5]}
    ser = nwp.DataFrame.from_dict(data, backend=backend).get_column(column)
    result = ser.hist(bin_count=bin_count, include_breakpoint=True)
    expected = {"breakpoint": expected_breakpoint, "count": expected_count}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("bins", [[1, 5, 10]])
def test_hist_bins_no_data(
    backend: EagerAllowed, bins: list[int], *, include_breakpoint: bool
) -> None:
    s = nwp.Series.from_iterable([], dtype=nw.Float64(), backend=backend)
    result = s.hist(bins, include_breakpoint=include_breakpoint)
    assert len(result) == 2
    assert result.get_column("count").sum() == 0


@pytest.mark.parametrize("bin_count", [1, 10])
def test_hist_bin_count_no_data(
    backend: EagerAllowed, bin_count: int, *, include_breakpoint: bool
) -> None:
    s = nwp.Series.from_iterable([], dtype=nw.Float64(), backend=backend)
    result = s.hist(bin_count=bin_count, include_breakpoint=include_breakpoint)
    assert len(result) == bin_count
    assert result.get_column("count").sum() == 0

    if include_breakpoint:
        bps = result.get_column("breakpoint").to_list()
        assert bps[0] == (1 / bin_count)
        if bin_count > 1:
            assert bps[-1] == 1


def test_hist_bins_none(backend: EagerAllowed) -> None:
    s = nwp.Series.from_iterable([1, 2, 3], backend=backend)
    result = s.hist(bins=None, bin_count=None)
    assert len(result) == 10


def test_hist_series_compat_flag(backend: EagerAllowed) -> None:
    # NOTE: Mainly for verifying `Expr.hist` has handled naming/collecting as struct
    # The flag itself is not desirable
    values = [1, 3, 8, 8, 2, 1, 3]
    s = nwp.Series.from_iterable(values, name="original", backend=backend)

    result = s.hist(
        bin_count=4,
        include_breakpoint=False,
        include_category=False,
        _compatibility_behavior="narwhals",
    )
    assert_equal_data(result, {"count": [3, 2, 0, 2]})

    result = s.hist(
        bin_count=4,
        include_breakpoint=False,
        include_category=False,
        _compatibility_behavior="polars",
    )
    assert_equal_data(result, {"original": [3, 2, 0, 2]})

    result = s.hist(bin_count=4, include_breakpoint=True, include_category=False)
    expected = {"breakpoint": [2.75, 4.5, 6.25, 8.0], "count": [3, 2, 0, 2]}
    assert_equal_data(result, expected)
