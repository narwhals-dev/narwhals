from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager
data = {"a": [1.0, 2.0, None, 4.0], "b": [None, 3.0, None, 5.0]}


def test_len(nw_eager_constructor: ConstructorEager) -> None:
    result = len(nw.from_native(nw_eager_constructor(data), eager_only=True))

    assert result == 4
