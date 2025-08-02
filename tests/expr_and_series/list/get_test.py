from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"str": [["a", "b"], [None, "b"]]}
expected = {"str": ["a", None]}


def test_get_expr(constructor: Constructor) -> None:
    result = nw.from_native(constructor(data)).select(nw.col("str").list.get(0))

    assert_equal_data(result, expected)


def test_get_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result = df["str"].list.get(0)
    assert_equal_data({"str": result}, expected)
