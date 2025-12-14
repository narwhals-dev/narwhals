from __future__ import annotations

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe


@pytest.mark.parametrize(
    ("characters", "expected"),
    [(None, ["foobar", "bar", "baz"]), ("foo", ["bar", "bar\n", " baz"])],
)
def test_str_strip_chars(characters: str | None, expected: list[str]) -> None:
    data = {"a": ["foobar", "bar\n", " baz"]}
    result = dataframe(data).select(nwp.col("a").str.strip_chars(characters))
    assert_equal_data(result, {"a": expected})
