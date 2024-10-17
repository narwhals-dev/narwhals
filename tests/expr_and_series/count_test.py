from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_count(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, None, 6], "z": [7.0, None, None]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b", "z").count())
    expected = {"a": [3], "b": [2], "z": [1]}
    assert_equal_data(result, expected)


def test_count_series(constructor_eager: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, None, 6], "z": [7.0, None, None]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {"a": [df["a"].count()], "b": [df["b"].count()], "z": [df["z"].count()]}
    expected = {"a": [3], "b": [2], "z": [1]}
    assert_equal_data(result, expected)
