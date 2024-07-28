from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_anyh(constructor_lazy: Any) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor_lazy(data))
    result = df.select(any=nw.any_horizontal(nw.col("a"), nw.col("b")))

    expected = {"any": [False, True, True]}
    compare_dicts(result, expected)
