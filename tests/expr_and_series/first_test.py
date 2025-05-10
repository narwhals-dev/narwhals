from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = {"a": [8, 2, 1], "b": [58, 5, 6], "c": [2.5, 1.0, 3.0]}


@pytest.mark.parametrize(("col", "expected"), [("a", 8), ("b", 58), ("c", 2.5)])
def test_first_series(
    constructor_eager: ConstructorEager, col: str, expected: float
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    result = series.first()
    assert_equal_data({col: [result]}, {col: [expected]})


def test_first_series_empty(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    series = series.filter(series > 50)
    result = series.first()
    assert result is None
