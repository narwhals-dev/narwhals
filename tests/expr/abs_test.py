from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_abs(constructor_with_pyarrow: Any) -> None:
    df = nw.from_native(
        constructor_with_pyarrow({"a": [1, 2, 3, -4, 5]}), eager_only=True
    )
    result = df.select(b=nw.col("a").abs())
    expected = {"b": [1, 2, 3, 4, 5]}
    compare_dicts(result, expected)
    result = df.select(b=df["a"].abs())
    compare_dicts(result, expected)
