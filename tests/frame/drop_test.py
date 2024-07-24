from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw


@pytest.mark.parametrize(
    ("drop", "left"),
    [
        (["a"], ["b", "z"]),
        (["a", "b"], ["z"]),
    ],
)
def test_drop(constructor: Any, drop: list[str], left: list[str]) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    assert df.drop(drop).columns == left
    assert df.drop(*drop).columns == left
