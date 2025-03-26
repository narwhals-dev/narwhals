from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_is_unique_expr(constructor: Constructor) -> None:
    data = {
        "a": [1, 1, 2],
        "b": [1, 2, 3],
        "index": [0, 1, 2],
    }
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b").is_unique(), "index").sort("index")
    expected = {
        "a": [False, False, True],
        "b": [True, True, True],
        "index": [0, 1, 2],
    }
    assert_equal_data(result, expected)


def test_is_unique_w_nulls_expr(constructor: Constructor) -> None:
    data = {
        "a": [None, 1, 2],
        "b": [None, 2, None],
        "index": [0, 1, 2],
    }
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b").is_unique(), "index").sort("index")
    expected = {
        "a": [True, True, True],
        "b": [False, True, False],
        "index": [0, 1, 2],
    }
    assert_equal_data(result, expected)


def test_is_unique_series(constructor_eager: ConstructorEager) -> None:
    data = {
        "a": [1, 1, 2],
        "b": [1, 2, 3],
        "index": [0, 1, 2],
    }
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.is_unique()
    expected = {
        "a": [False, False, True],
    }
    assert_equal_data({"a": result}, expected)
