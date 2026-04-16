from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [1.0, None, None, 3.0], "b": [1.0, None, 4.0, 5.0]}


def test_null_count_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b").null_count())
    expected = {"a": [2], "b": [1]}
    assert_equal_data(result, expected)


def test_null_count_series(constructor_eager: ConstructorEager) -> None:
    data = [1, 2, None]
    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    result = series.null_count()
    assert result == 1
