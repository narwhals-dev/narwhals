from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": ["foobar", "bar\n", " baz"]}


@pytest.mark.parametrize(
    ("characters", "expected"),
    [(None, {"a": ["foobar", "bar", "baz"]}), ("foo", {"a": ["bar", "bar\n", " baz"]})],
)
def test_str_strip_chars(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    characters: str | None,
    expected: Any,
) -> None:
    if "ibis" in str(constructor) and characters is not None:
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("a").str.strip_chars(characters))
    assert_equal_data(result_frame, expected)


@pytest.mark.parametrize(
    ("characters", "expected"),
    [(None, {"a": ["foobar", "bar", "baz"]}), ("foo", {"a": ["bar", "bar\n", " baz"]})],
)
def test_str_strip_chars_series(
    constructor_eager: ConstructorEager, characters: str | None, expected: Any
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.strip_chars(characters)
    assert_equal_data({"a": result_series}, expected)
