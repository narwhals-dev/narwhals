from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": [1.0, None, None, 3.0],
    "b": [1.0, None, 4, 5.0],
}


def test_n_unique(constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(nw.all().n_unique())
    expected = {
        "a": [3],
        "b": [4],
    }
    compare_dicts(result, expected)
    assert df["a"].n_unique() == 3
    assert df["b"].n_unique() == 4
    compare_dicts(result, expected)
