from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

def test_series_sum(constructor: Any) -> None:
    data = {
        "a": [0, 1, 2, 3, 4],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, None, 2, 1],
    }
    df = nw.from_native(constructor(data), eager_only=True, allow_series=True)


    result = df.select(nw.col("a", "b", "c").sum())
    result_native = nw.to_native(result)
        # Extract the sum values from the resulting DataFrame
    result_dict = {
        "a": [result_native["a"][0]],
        "b": [result_native["b"][0]],
        "c": [result_native["c"][0]]
    }
    expected = {"a": [10], "b": [14], "c": [12]}
    compare_dicts(result_dict, expected)