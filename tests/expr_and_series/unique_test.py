from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 1, 2]}
data_str = {"a": ["x", "x", "y"]}


def test_unique_expr(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").unique())
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)


def test_unique_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data_str), eager_only=True)["a"]
    result = series.unique(maintain_order=True)
    expected = {"a": ["x", "y"]}
    assert_equal_data({"a": result}, expected)
