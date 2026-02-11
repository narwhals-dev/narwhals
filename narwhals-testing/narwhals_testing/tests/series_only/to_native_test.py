from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = [4, 4, 4, 1, 6, 6, 4, 4, 1, 1]


def test_to_native(constructor_eager: ConstructorEager) -> None:
    orig_series = constructor_eager({"a": data})["a"]  # type: ignore[index]
    nw_series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    result = nw_series.to_native()
    assert isinstance(result, orig_series.__class__)
