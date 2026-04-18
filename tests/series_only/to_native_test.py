from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = [4, 4, 4, 1, 6, 6, 4, 4, 1, 1]


def test_to_native(nw_eager_constructor: ConstructorEager) -> None:
    orig_series = nw_eager_constructor({"a": data})["a"]  # type: ignore[index]
    nw_series = nw.from_native(nw_eager_constructor({"a": data}), eager_only=True)["a"]
    result = nw_series.to_native()
    assert isinstance(result, orig_series.__class__)
