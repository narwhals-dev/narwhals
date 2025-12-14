from __future__ import annotations

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe


@pytest.mark.parametrize(
    ("by", "expected"),
    [
        ("_", [["foo bar"], ["foo", "bar"], ["foo", "bar", "baz"], ["foo,bar"]]),
        (",", [["foo bar"], ["foo_bar"], ["foo_bar_baz"], ["foo", "bar"]]),
    ],
)
def test_str_split(by: str, expected: list[list[str]]) -> None:
    data = {"a": ["foo bar", "foo_bar", "foo_bar_baz", "foo,bar"]}
    result = dataframe(data).select(nwp.col("a").str.split(by))
    assert_equal_data(result, {"a": expected})
