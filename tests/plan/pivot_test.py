from __future__ import annotations

# ruff: noqa: FBT001
from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING, Any

import pytest

from narwhals.exceptions import NarwhalsError
from tests.plan.utils import assert_equal_data, dataframe, re_compile

if TYPE_CHECKING:
    from narwhals.typing import PivotAgg
    from tests.conftest import Data


XFAIL_NOT_IMPL_AGG = pytest.mark.xfail(
    reason="TODO: `ArrowDataFrame.pivot_agg`", raises=NotImplementedError
)
XFAIL_NOT_IMPL_ON_MULTIPLE = pytest.mark.xfail(
    reason="TODO: `ArrowDataFrame.pivot(on: list[str])`", raises=NotImplementedError
)


@pytest.fixture(scope="module")
def scores() -> Data:
    """Ripped from `pl.DataFrame.pivot` docstring."""
    return {
        "name": ["Cady", "Cady", "Karen", "Karen"],
        "subject": ["maths", "physics", "maths", "physics"],
        "test_1": [98, 99, 61, 58],
        "test_2": [100, 100, 60, 60],
    }


data = {
    "ix": [1, 2, 1, 1, 2, 2],
    "iy": [1, 2, 2, 1, 2, 1],
    "col": ["b", "b", "a", "a", "a", "a"],
    "col_b": ["x", "y", "x", "y", "x", "y"],
    "foo": [7, 1, 0, 1, 2, 2],
    "bar": [9, 4, 0, 2, 0, 0],
}

data_no_dups = {
    "ix": [1, 1, 2, 2],
    "col": ["a", "b", "a", "b"],
    "foo": [1, 2, 3, 4],
    "bar": ["x", "y", "z", "w"],
}

data_no_dups_unordered = {
    "ix": [1, 1, 2, 2],
    "col": ["b", "a", "b", "a"],
    "foo": [1, 2, 3, 4],
    "bar": ["x", "y", "z", "w"],
}
"""Variant to give the same order as `data["col"]`, but without needing an aggregate."""


# NOTE: `tests::frame::pivot_test.py::test_pivot`
@XFAIL_NOT_IMPL_AGG
@pytest.mark.parametrize(
    ("agg_func", "expected"),
    [
        (
            "min",
            {
                "ix": [1, 2],
                "foo_a": [0, 2],
                "foo_b": [7, 1],
                "bar_a": [0, 0],
                "bar_b": [9, 4],
            },
        ),
        (
            "max",
            {
                "ix": [1, 2],
                "foo_a": [1, 2],
                "foo_b": [7, 1],
                "bar_a": [2, 0],
                "bar_b": [9, 4],
            },
        ),
        (
            "first",
            {
                "ix": [1, 2],
                "foo_a": [0, 2],
                "foo_b": [7, 1],
                "bar_a": [0, 0],
                "bar_b": [9, 4],
            },
        ),
        (
            "last",
            {
                "ix": [1, 2],
                "foo_a": [1, 2],
                "foo_b": [7, 1],
                "bar_a": [2, 0],
                "bar_b": [9, 4],
            },
        ),
        (
            "sum",
            {
                "ix": [1, 2],
                "foo_a": [1, 4],
                "foo_b": [7, 1],
                "bar_a": [2, 0],
                "bar_b": [9, 4],
            },
        ),
        (
            "mean",
            {
                "ix": [1, 2],
                "foo_a": [0.5, 2.0],
                "foo_b": [7.0, 1.0],
                "bar_a": [1.0, 0.0],
                "bar_b": [9.0, 4.0],
            },
        ),
        (
            "median",
            {
                "ix": [1, 2],
                "foo_a": [0.5, 2.0],
                "foo_b": [7.0, 1.0],
                "bar_a": [1.0, 0.0],
                "bar_b": [9.0, 4.0],
            },
        ),
        (
            "len",
            {
                "ix": [1, 2],
                "foo_a": [2, 2],
                "foo_b": [1, 1],
                "bar_a": [2, 2],
                "bar_b": [1, 1],
            },
        ),
    ],
)
@pytest.mark.parametrize(("on", "index"), [("col", "ix"), (["col"], ["ix"])])
def test_pivot_agg(
    on: str | list[str], index: str | list[str], agg_func: PivotAgg, expected: Data
) -> None:
    df = dataframe(data)
    result = df.pivot(
        on=on,
        index=index,
        values=["foo", "bar"],
        aggregate_function=agg_func,
        sort_columns=True,
    )

    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("sort_columns", "expected"),
    [
        (True, ["ix", "foo_a", "foo_b", "bar_a", "bar_b"]),
        (False, ["ix", "foo_b", "foo_a", "bar_b", "bar_a"]),
    ],
)
def test_pivot_sort_columns(sort_columns: bool, expected: list[str]) -> None:
    df = dataframe(data_no_dups_unordered)
    result = df.pivot(
        on="col", index="ix", values=["foo", "bar"], sort_columns=sort_columns
    )
    assert result.columns == expected


@XFAIL_NOT_IMPL_AGG
@pytest.mark.parametrize(
    ("on", "values", "expected"),
    [
        (["col"], ["foo"], ["ix", "b", "a"]),
        (["col"], ["foo", "bar"], ["ix", "foo_b", "foo_a", "bar_b", "bar_a"]),
        pytest.param(
            ["col", "col_b"],
            ["foo"],
            ["ix", '{"b","x"}', '{"b","y"}', '{"a","x"}', '{"a","y"}'],
            marks=XFAIL_NOT_IMPL_ON_MULTIPLE,
        ),
        pytest.param(
            ["col", "col_b"],
            ["foo", "bar"],
            [
                "ix",
                'foo_{"b","x"}',
                'foo_{"b","y"}',
                'foo_{"a","x"}',
                'foo_{"a","y"}',
                'bar_{"b","x"}',
                'bar_{"b","y"}',
                'bar_{"a","x"}',
                'bar_{"a","y"}',
            ],
            marks=XFAIL_NOT_IMPL_ON_MULTIPLE,
        ),
    ],
)
def test_pivot_names_out(
    on: list[str], values: list[str], expected: list[str]
) -> None:  # pragma: no cover
    df = dataframe(data)
    result = df.pivot(on=on, values=values, aggregate_function="min", index="ix").columns
    assert result == expected


@pytest.mark.parametrize(
    ("data_", "context"),
    [
        (data_no_dups, does_not_raise()),
        (data, pytest.raises((ValueError, NarwhalsError))),
    ],
    ids=["no-duplicates", "duplicated"],
)
def test_pivot_no_agg(data_: Data, context: Any) -> None:
    expected_no_dups = {
        "ix": [1, 2],
        "foo_a": [1, 3],
        "foo_b": [2, 4],
        "bar_a": ["x", "z"],
        "bar_b": ["y", "w"],
    }
    df = dataframe(data_)
    with context:
        result = df.pivot("col", index="ix")
        assert_equal_data(result, expected_no_dups)


def test_pivot_no_index_no_values() -> None:
    df = dataframe(data_no_dups)
    with pytest.raises(
        ValueError, match=re_compile(r"at least one of.+values.+index.+must")
    ):
        df.pivot(on="col")


# NOTE: `tests::frame::pivot_test.py::test_pivot_no_index`
def test_pivot_implicit_index() -> None:
    inferred_index_names = "ix", "bar"
    df = dataframe(data_no_dups)
    result = df.pivot(on="col", values="foo").sort(inferred_index_names)
    expected = {
        "ix": [1, 1, 2, 2],
        "bar": ["x", "y", "w", "z"],
        "a": [1.0, None, None, 3.0],
        "b": [None, 2.0, 4.0, None],
    }
    assert_equal_data(result, expected)


def test_pivot_test_scores(scores: Data) -> None:
    df = dataframe(scores)
    expected = {"name": ["Cady", "Karen"], "maths": [98, 61], "physics": [99, 58]}
    result = df.pivot("subject", index="name", values="test_1")
    assert_equal_data(result, expected)
    result = df.pivot(
        "subject", on_columns=["maths", "physics"], index="name", values="test_1"
    )
    assert_equal_data(result, expected)
