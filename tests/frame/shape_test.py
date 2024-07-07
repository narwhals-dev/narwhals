from typing import Any

import narwhals.stable.v1 as nw


def test_shape(constructor_with_pyarrow: Any) -> None:
    result = nw.from_native(
        constructor_with_pyarrow({"a": [1, 2], "b": [4, 5], "c": [7, 8]}), eager_only=True
    ).shape
    expected = (2, 3)
    assert result == expected
