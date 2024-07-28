from typing import Any

import narwhals.stable.v1 as nw


def test_shape(constructor: Any) -> None:
    result = nw.from_native(constructor({"a": [1, 2]}), eager_only=True)["a"].shape
    expected = (2,)
    assert result == expected
