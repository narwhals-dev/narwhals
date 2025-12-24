from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import Any

import pytest

from narwhals.exceptions import NarwhalsError
from tests.plan.utils import assert_equal_data, dataframe, re_compile

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

XFAIL_VALUES_MULTIPLE = pytest.mark.xfail(
    reason="TODO: `ArrowDataFrame.pivot(values=(..., ...))`", raises=NotImplementedError
)


@XFAIL_VALUES_MULTIPLE
@pytest.mark.parametrize(
    ("data_", "context"),
    [
        (data_no_dups, does_not_raise()),
        (data, pytest.raises((ValueError, NarwhalsError))),
    ],
)
def test_pivot_no_agg(data_: Any, context: Any) -> None:
    df = dataframe(data_)
    with context:
        df.pivot("col", index="ix")


def test_pivot_no_index_no_values() -> None:
    df = dataframe(data_no_dups)
    with pytest.raises(
        ValueError, match=re_compile(r"at least one of.+values.+index.+must")
    ):
        df.pivot(on="col")


@pytest.mark.xfail(
    reason="BUG: 'pyarrow.lib.DataType' object has no attribute 'fields'. Did you mean: 'field'?",
    raises=AttributeError,
)
def test_pivot_no_index() -> None:  # pragma: no cover
    df = dataframe(data_no_dups)
    result = df.pivot(on="col", values="foo").sort("ix", "bar")
    expected = {
        "ix": [1, 1, 2, 2],
        "bar": ["x", "y", "w", "z"],
        "a": [1.0, None, None, 3.0],
        "b": [None, 2.0, 4.0, None],
    }
    assert_equal_data(result, expected)
