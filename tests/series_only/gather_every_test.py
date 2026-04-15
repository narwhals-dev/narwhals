from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.testing.typing import ConstructorEager

data = {"a": list(range(10))}


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_gather_every_series(
    constructor_eager: ConstructorEager, n: int, offset: int
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]

    result = series.gather_every(n=n, offset=offset)
    expected = data["a"][offset::n]

    assert_equal_data({"a": result}, {"a": expected})
