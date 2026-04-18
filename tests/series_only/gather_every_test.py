from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

data = {"a": list(range(10))}


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_gather_every_series(
    nw_eager_constructor: ConstructorEager, n: int, offset: int
) -> None:
    series = nw.from_native(nw_eager_constructor(data), eager_only=True)["a"]

    result = series.gather_every(n=n, offset=offset)
    expected = data["a"][offset::n]

    assert_equal_data({"a": result}, {"a": expected})
