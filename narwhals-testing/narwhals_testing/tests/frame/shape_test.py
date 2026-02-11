from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_shape(constructor_eager: ConstructorEager) -> None:
    result = nw.from_native(
        constructor_eager({"a": [1, 2], "b": [4, 5], "c": [7, 8]}), eager_only=True
    ).shape
    expected = (2, 3)
    assert result == expected
