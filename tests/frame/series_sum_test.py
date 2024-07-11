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


    # Test sum of series "a"
    result = df.select(nw.col("c").sum())
    result_native = nw.to_native(result)
    expected = sum(x for x in data["c"] if x is not None)
    assert result_native.to_numpy()[0, 0] == expected