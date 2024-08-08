from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw


@pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "expected"),
    [
        ({"a": ["123abc", "abc456"]}, r"abc\b", "ABC", 1, {"a": ["123ABC", "abc456"]}),
        ({"a": ["abc abc", "abc456"]}, r"abc", "", 1, {"a": [" abc", "456"]}),
        ({"a": ["abc abc abc", "456abc"]}, r"abc", "", -1, {"a": ["  ", "456"]}),
    ],
)
def test_str_replace_series(
    constructor_eager: Any,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    n: int,
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.replace(pattern=pattern, value=value, n=n)
    assert result_series.to_list() == expected["a"]


@pytest.mark.parametrize(
    ("data", "pattern", "value", "expected"),
    [
        ({"a": ["123abc", "abc456"]}, r"abc\b", "ABC", {"a": ["123ABC", "abc456"]}),
        ({"a": ["abc abc", "abc456"]}, r"abc", "", {"a": [" ", "456"]}),
        ({"a": ["abc abc abc", "456abc"]}, r"abc", "", {"a": ["  ", "456"]}),
    ],
)
def test_str_replace_all_series(
    constructor_eager: Any,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.replace_all(pattern=pattern, value=value)
    assert result_series.to_list() == expected["a"]
