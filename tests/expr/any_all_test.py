from typing import Any

import narwhals as nw
from tests.utils import compare_dicts


def test_any_all(constructor: Any) -> None:
    df = nw.from_native(
        constructor(
            {
                "a": [True, False, True],
                "b": [True, True, True],
                "c": [False, False, False],
            }
        )
    )
    result = nw.to_native(df.select(nw.all().all()))
    expected = {"a": [False], "b": [True], "c": [False]}
    compare_dicts(result, expected)
    result = nw.to_native(df.select(nw.all().any()))
    expected = {"a": [True], "b": [True], "c": [False]}
    compare_dicts(result, expected)
