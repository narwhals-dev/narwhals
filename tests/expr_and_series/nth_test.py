from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]}


@pytest.mark.parametrize(
    ("idx", "expected"),
    [
        (0, {"a": [1, 3, 2]}),
        ([0, 1], {"a": [1, 3, 2], "b": [4, 4, 6]}),
        ([0, 2], {"a": [1, 3, 2], "z": [7.1, 8, 9]}),
    ],
)
def test_nth(
    constructor: Constructor, idx: int | list[int], expected: dict[str, list[int]]
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.nth(idx))
    compare_dicts(result, expected)
