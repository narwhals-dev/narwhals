from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import compare_dicts

data = {
    "a": [1, 1, 2, 3, 2],
    "b": [1, 2, 3, 2, 1],
}


def test_is_first_distinct_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.all().is_first_distinct())
    expected = {
        "a": [True, False, True, True, False],
        "b": [True, True, True, False, False],
    }
    compare_dicts(result, expected)


def test_is_first_distinct_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.is_first_distinct()
    expected = {
        "a": [True, False, True, True, False],
    }
    compare_dicts({"a": result}, expected)
