from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": [0, 1, 2, 3, 4],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
}


def test_cum_sum_simple(constructor_with_pyarrow: Any) -> None:
    df = nw.from_native(constructor_with_pyarrow(data), eager_only=True)
    result = df.select(nw.all().cum_sum())
    expected = {
        "a": [0, 1, 3, 6, 10],
        "b": [1, 3, 6, 11, 14],
        "c": [5, 9, 12, 14, 15],
    }
    compare_dicts(result, expected)
    result = df.select(
        df["a"].cum_sum(),
        df["b"].cum_sum(),
        df["c"].cum_sum(),
    )
    compare_dicts(result, expected)
