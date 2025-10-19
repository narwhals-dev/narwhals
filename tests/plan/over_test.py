from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("pyarrow")

import narwhals as nw
import narwhals._plan as nwp
from narwhals._plan import selectors as ncs
from narwhals.exceptions import InvalidOperationError
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from narwhals._plan.typing import IntoExprColumn, OneOrIterable
    from tests.conftest import Data


@pytest.fixture
def data() -> Data:
    return {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "i": [0, 1, 2, 3, 4],
    }


XFAIL_NO_CUM_SUM = pytest.mark.xfail(
    reason="Not implemented `CompliantExpr.cum_sum`", raises=NotImplementedError
)


@pytest.mark.parametrize(
    "partition_by",
    [
        "a",
        ["a"],
        nwp.nth(0),
        ncs.string(),
        ncs.by_dtype(nw.String),
        ncs.by_name("a"),
        ncs.matches(r"a"),
        ncs.all() - ncs.numeric(),
    ],
)
def test_over_single(data: Data, partition_by: OneOrIterable[IntoExprColumn]) -> None:
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "i": [0, 1, 2, 3, 4],
        "c_max": [5, 5, 3, 3, 3],
    }
    result = (
        dataframe(data)
        .with_columns(c_max=nwp.col("c").max().over(partition_by))
        .sort("i")
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    "partition_by",
    [
        ("a", "b"),
        [nwp.col("a"), nwp.col("b")],
        [nwp.nth(0), nwp.nth(1)],
        nwp.col("a", "b"),
        nwp.nth(0, 1),
        ncs.by_name("a", "b"),
        ncs.matches(r"a|b"),
        ncs.all() - ncs.by_name(["c", "i"]),
    ],
    ids=[
        "tuple[str]",
        "col-col",
        "nth-nth",
        "cols",
        "index_columns",
        "by_name",
        "matches",
        "binary_selector",
    ],
)
def test_over_multiple(data: Data, partition_by: OneOrIterable[IntoExprColumn]) -> None:
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "i": [0, 1, 2, 3, 4],
        "c_min": [5, 4, 1, 2, 1],
    }
    result = (
        dataframe(data)
        .with_columns(c_min=nwp.col("c").min().over(partition_by))
        .sort("i")
    )
    assert_equal_data(result, expected)


def test_over_std_var(data: Data) -> None:
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "i": [0, 1, 2, 3, 4],
        "c_std0": [0.5, 0.5, 0.816496580927726, 0.816496580927726, 0.816496580927726],
        "c_std1": [0.7071067811865476, 0.7071067811865476, 1.0, 1.0, 1.0],
        "c_var0": [
            0.25,
            0.25,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
        ],
        "c_var1": [0.5, 0.5, 1.0, 1.0, 1.0],
    }

    result = (
        dataframe(data)
        .with_columns(
            c_std0=nwp.col("c").std(ddof=0).over("a"),
            c_std1=nwp.col("c").std(ddof=1).over("a"),
            c_var0=nwp.col("c").var(ddof=0).over("a"),
            c_var1=nwp.col("c").var(ddof=1).over("a"),
        )
        .sort("i")
    )
    assert_equal_data(result, expected)


# NOTE: Supporting this for pyarrow is new 🥳
def test_over_anonymous_reduction() -> None:
    df = dataframe({"a": [1, 1, 2], "b": [4, 5, 6]})
    result = df.with_columns(nwp.all().sum().over("a").name.suffix("_sum")).sort("a", "b")
    expected = {"a": [1, 1, 2], "b": [4, 5, 6], "a_sum": [2, 2, 2], "b_sum": [9, 9, 6]}
    assert_equal_data(result, expected)


def test_over_raise_len_change(data: Data) -> None:
    df = dataframe(data)
    with pytest.raises(InvalidOperationError):
        df.select(nwp.col("b").drop_nulls().over("a"))


# NOTE: Currently raising `InvalidOperationError: `cum_sum()` is not supported in a `group_by` context`
# (main): https://github.com/narwhals-dev/narwhals/blob/ecde261d799a711c2e0a7acf11b108bc45035dc9/narwhals/_arrow/expr.py#L116-L118
# NotImplementedError: Only aggregation or literal operations are supported in grouped `over` context for PyArrow.
@pytest.mark.xfail(reason="Not implemented `cum_sum`", raises=InvalidOperationError)
def test_unsupported_over(data: Data) -> None:
    df = dataframe(data)
    with pytest.raises(NotImplementedError):
        df.select(nwp.col("a").shift(1).cum_sum().over("b"))


@XFAIL_NO_CUM_SUM
def test_over_without_partition_by() -> None:  # pragma: no cover
    df = dataframe({"a": [1, -1, 2], "i": [0, 2, 1]})
    result = (
        df.with_columns(b=nwp.col("a").abs().cum_sum().over(order_by="i"))
        .sort("i")
        .select("a", "b", "i")
    )
    expected = {"a": [1, 2, -1], "b": [1, 3, 4], "i": [0, 1, 2]}
    assert_equal_data(result, expected)


def test_aggregation_over_without_partition_by() -> None:
    df = dataframe({"a": [1, -1, 2], "i": [0, 2, 1]})
    result = (
        df.with_columns(b=nwp.col("a").diff().sum().over(order_by="i"))
        .sort("i")
        .select("a", "b", "i")
    )
    expected = {"a": [1, 2, -1], "b": [-2, -2, -2], "i": [0, 1, 2]}
    assert_equal_data(result, expected)


def test_len_over_2369() -> None:
    df = dataframe({"a": [1, 2, 4], "b": ["x", "x", "y"]})
    result = df.with_columns(a_len_per_group=nwp.len().over("b")).sort("a")
    expected = {"a": [1, 2, 4], "b": ["x", "x", "y"], "a_len_per_group": [2, 2, 1]}
    assert_equal_data(result, expected)
