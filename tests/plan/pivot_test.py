from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING, Any

import pytest

from narwhals.exceptions import NarwhalsError
from tests.plan.utils import assert_equal_data, dataframe, re_compile

if TYPE_CHECKING:
    from tests.conftest import Data


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


@pytest.mark.parametrize(
    ("data_", "context"),
    [
        (data_no_dups, does_not_raise()),
        (data, pytest.raises((ValueError, NarwhalsError))),
    ],
    ids=["no-duplicates", "duplicated"],
)
def test_pivot_no_agg(data_: Any, context: Any) -> None:
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
