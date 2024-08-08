from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw


@pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"),
    [
        (
            {"a": ["123abc", "abc456"]},
            r"abc\b",
            "ABC",
            1,
            False,
            {"a": ["123ABC", "abc456"]},
        ),
        ({"a": ["abc abc", "abc456"]}, r"abc", "", 1, False, {"a": [" abc", "456"]}),
        ({"a": ["abc abc abc", "456abc"]}, r"abc", "", -1, False, {"a": ["  ", "456"]}),
        (
            {"a": ["Dollar $ign", "literal"]},
            r"$",
            "S",
            -1,
            True,
            {"a": ["Dollar Sign", "literal"]},
        ),
    ],
)
def test_str_replace_series(
    constructor_eager: Any,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    n: int,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.replace(
        pattern=pattern, value=value, n=n, literal=literal
    )
    assert result_series.to_list() == expected["a"]


@pytest.mark.parametrize(
    ("data", "pattern", "value", "literal", "expected"),
    [
        (
            {"a": ["123abc", "abc456"]},
            r"abc\b",
            "ABC",
            False,
            {"a": ["123ABC", "abc456"]},
        ),
        ({"a": ["abc abc", "abc456"]}, r"abc", "", False, {"a": [" ", "456"]}),
        ({"a": ["abc abc abc", "456abc"]}, r"abc", "", False, {"a": ["  ", "456"]}),
        (
            {"a": ["Dollar $ign", "literal"]},
            r"$",
            "S",
            True,
            {"a": ["Dollar Sign", "literal"]},
        ),
    ],
)
def test_str_replace_all_series(
    constructor_eager: Any,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.replace_all(pattern=pattern, value=value, literal=literal)
    assert result_series.to_list() == expected["a"]
