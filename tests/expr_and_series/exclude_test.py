from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor


@pytest.mark.parametrize(
    ("exclude_selector", "expected_cols"),
    [
        (nw.exclude("a"), ["b", "z"]),
        (nw.exclude("b", "z"), ["a"]),
        (nw.exclude(["a"]), ["b", "z"]),
        (nw.exclude(["b", "z"]), ["a"]),
    ],
)
def test_exclude(
    constructor: Constructor, exclude_selector: nw.Expr, expected_cols: list[str]
) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}

    df = nw.from_native(constructor(data))
    result = df.select(exclude_selector)

    expected = {col: data[col] for col in expected_cols}
    assert_equal_data(result, expected)
