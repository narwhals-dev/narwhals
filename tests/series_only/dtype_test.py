from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = {"a": [1, 3, 2]}


def test_dtype(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.dtype
    assert result == nw.Int64
    assert result.is_numeric()
