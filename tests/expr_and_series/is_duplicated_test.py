from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": [1, 1, 2],
    "b": [1, 2, 3],
}


def test_is_duplicated_expr(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.all().is_duplicated())
    expected = {
        "a": [True, True, False],
        "b": [False, False, False],
    }
    compare_dicts(result, expected)


def test_is_duplicated_series(constructor_eager: Any) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.is_duplicated()
    expected = {"a": [True, True, False]}
    compare_dicts({"a": result}, expected)
