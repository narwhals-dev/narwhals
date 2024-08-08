from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw


@pytest.mark.parametrize(
    ("data", "pattern", "value", "expected"),
    [
        ({"a": ["123abc", "abc456"]}, r"abc\b", "ABC", {"a": ["123ABC", "abc456"]}),
    ],
)
def test_str_replace_series(
    constructor_eager: Any,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.replace(pattern=pattern, value=value)
    assert result_series.to_list() == expected["a"]
