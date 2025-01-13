from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import Constructor


def test_order_dependent_raises_in_lazy(constructor: Constructor) -> None:
    lf = nw.from_native(constructor({"a": [1, 2, 3]})).lazy()
    with pytest.raises(TypeError, match="Order-dependent expressions"):
        lf.select(nw.col("a").diff())
