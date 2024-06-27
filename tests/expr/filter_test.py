from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "i": [0, 1, 2, 3, 4],
    "a": [0, 1, 2, 3, 4],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
}


def test_filter(constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(nw.col("a").filter(nw.col("i") < 2, nw.col("c") == 5))
    expected = {
        "a": [0],
    }
    compare_dicts(result, expected)
    result = df.select(df["a"].filter((df["i"] < 2) & (df["c"] == 5)))
    compare_dicts(result, expected)
