from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import compare_dicts

data = {"a": [1, 1, 2], "b": [1, 2, 3], "index": [0, 1, 2]}


def test_is_duplicated_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b").is_duplicated(), "index").sort("index")
    expected = {"a": [True, True, False], "b": [False, False, False], "index": [0, 1, 2]}
    compare_dicts(result, expected)


def test_is_duplicated_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.is_duplicated()
    expected = {"a": [True, True, False]}
    compare_dicts({"a": result}, expected)
