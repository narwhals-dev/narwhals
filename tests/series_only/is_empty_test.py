from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_is_empty(nw_eager_constructor: ConstructorEager) -> None:
    series = nw.from_native(nw_eager_constructor({"a": [1, 2, 3]}), eager_only=True)["a"]
    assert not series.is_empty()
    assert not series[:1].is_empty()
    assert len(series[:1]) == 1
    assert series[:0].is_empty()
