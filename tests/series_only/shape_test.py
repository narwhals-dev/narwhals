from typing import Any

import narwhals.stable.v1 as nw


def test_shape(constructor_series: Any) -> None:
    result = nw.from_native(constructor_series([1, 2]), series_only=True).shape
    expected = (2,)
    assert result == expected
