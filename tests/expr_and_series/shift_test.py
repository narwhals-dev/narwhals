from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "i": [0, 1, 2, 3, 4],
    "a": [0, 1, 2, 3, 4],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
}


def test_shift(request: Any, constructor_lazy: Any) -> None:
    if "pyarrow_table" in str(constructor_lazy):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_lazy(data), eager_only=True)
    result = df.with_columns(nw.col("a", "b", "c").shift(2)).filter(nw.col("i") > 1)
    expected = {
        "i": [2, 3, 4],
        "a": [0, 1, 2],
        "b": [1, 2, 3],
        "c": [5, 4, 3],
    }
    compare_dicts(result, expected)
    result = df.select(
        df["i"],
        df["a"].shift(2),
        df["b"].shift(2),
        df["c"].shift(2),
    ).filter(nw.col("i") > 1)
    compare_dicts(result, expected)
