from __future__ import annotations

import pytest

import narwhals as nw_main
import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": list(range(10))}


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_gather_every_expr(
    constructor_eager: ConstructorEager, n: int, offset: int
) -> None:
    df = nw.from_native(constructor_eager(data))

    result = df.select(nw.col("a").gather_every(n=n, offset=offset))
    expected = {"a": data["a"][offset::n]}

    assert_equal_data(result, expected)

    with pytest.deprecated_call():
        df.select(nw_main.col("a").gather_every(n=n, offset=offset))


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_gather_every_series(
    constructor_eager: ConstructorEager, n: int, offset: int
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]

    result = series.gather_every(n=n, offset=offset)
    expected = data["a"][offset::n]

    assert_equal_data({"a": result}, {"a": expected})
