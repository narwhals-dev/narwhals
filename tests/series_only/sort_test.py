from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


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
    series = nw.from_native(constructor_eager({"a": [1, 3, 2, None]}), eager_only=True)[
        "a"
    ]
    result = series.sort(descending=descending, nulls_last=nulls_last)

    assert_equal_data({"a": result}, {"a": expected})
