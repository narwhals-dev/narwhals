from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Mapping

data: Mapping[str, Any] = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]}


@pytest.mark.parametrize(
    ("idx", "expected"),
    [
        (0, {"a": [1, 3, 2]}),
        ([0, 1], {"a": [1, 3, 2], "b": [4, 4, 6]}),
        ([0, 2], {"a": [1, 3, 2], "z": [7.1, 8.0, 9.0]}),
    ],
)
def test_nth(
    constructor: Constructor, idx: int | list[int], expected: dict[str, list[int]]
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.nth(idx))
    assert_equal_data(result, expected)
