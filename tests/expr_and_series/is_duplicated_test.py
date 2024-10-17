from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"a": [1, 1, 2], "b": [1, 2, 3], "index": [0, 1, 2]}


def test_is_duplicated_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b").is_duplicated(), "index").sort("index")
    expected = {"a": [True, True, False], "b": [False, False, False], "index": [0, 1, 2]}
    assert_equal_data(result, expected)


def test_is_duplicated_series(constructor_eager: Any) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.is_duplicated()
    expected = {"a": [True, True, False]}
    assert_equal_data({"a": result}, expected)
