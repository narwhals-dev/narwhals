from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_allh(constructor_lazy: Any) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor_lazy(data))
    result = df.select(all=nw.all_horizontal(nw.col("a"), nw.col("b")))

    expected = {"all": [False, False, True]}
    compare_dicts(result, expected)


def test_allh_series(constructor: Any) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(all=nw.all_horizontal(df["a"], df["b"]))

    expected = {"all": [False, False, True]}
    compare_dicts(result, expected)
