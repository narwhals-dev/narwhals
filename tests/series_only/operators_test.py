from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("__eq__", [False, True, False]),
        ("__ne__", [True, False, True]),
        ("__le__", [True, True, False]),
        ("__lt__", [True, False, False]),
        ("__ge__", [False, True, True]),
        ("__gt__", [False, False, True]),
    ],
)
def test_comparand_operators(
    constructor_eager: Any, operator: str, expected: list[bool]
) -> None:
    data = [0, 1, 2]
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    result = getattr(s, operator)(1)
    assert result.to_list() == expected


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("__and__", [True, False, False, False]),
        ("__or__", [True, True, True, False]),
    ],
)
def test_logic_operators(
    constructor_eager: Any, operator: str, expected: list[bool]
) -> None:
    data = [True, True, False, False]
    other_data = [True, False, True, False]
    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    other = nw.from_native(constructor_eager({"a": other_data}), eager_only=True)["a"]
    result = getattr(series, operator)(other)
    assert result.to_list() == expected
