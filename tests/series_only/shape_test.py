from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_shape(nw_eager_constructor: ConstructorEager) -> None:
    result = nw.from_native(nw_eager_constructor({"a": [1, 2]}), eager_only=True)[
        "a"
    ].shape
    expected = (2,)
    assert result == expected
