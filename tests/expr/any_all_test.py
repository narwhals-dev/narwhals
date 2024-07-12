from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_any_all(constructor_with_pyarrow: Any) -> None:
    df = nw.from_native(
        constructor_with_pyarrow(
            {
                "a": [True, False, True],
                "b": [True, True, True],
                "c": [False, False, False],
            }
        )
    )
    result = df.select(nw.all().all())
    expected = {"a": [False], "b": [True], "c": [False]}
    compare_dicts(result, expected)
    result = df.select(nw.all().any())
    expected = {"a": [True], "b": [True], "c": [False]}
    compare_dicts(result, expected)
