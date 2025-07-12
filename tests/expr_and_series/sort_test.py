from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals.stable.v1 as nw_v1

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = {"a": [0, 0, 2, -1], "b": [1, 3, 2, None]}


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, [3, 2, 1, None]),
        (True, False, [None, 3, 2, 1]),
        (False, True, [1, 2, 3, None]),
        (False, False, [None, 1, 2, 3]),
    ],
)
def test_sort_series(
    constructor_eager: ConstructorEager, descending: Any, nulls_last: Any, expected: Any
) -> None:
    series = nw_v1.from_native(constructor_eager(data), eager_only=True)["b"]
    result = series.sort(descending=descending, nulls_last=nulls_last)
    assert (
        result
        == nw_v1.from_native(constructor_eager({"a": expected}), eager_only=True)["a"]
    )
