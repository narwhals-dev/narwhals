from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = [1, 4, 2, 5]


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("left", [True, True, True, False]),
        ("right", [False, True, True, True]),
        ("both", [True, True, True, True]),
        ("none", [False, True, True, False]),
    ],
)
def test_is_between(constructor_eager: Any, closed: str, expected: list[bool]) -> None:
    ser = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    result = ser.is_between(1, 5, closed=closed)
    compare_dicts({"a": result}, {"a": expected})
