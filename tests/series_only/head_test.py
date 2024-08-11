from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw


@pytest.mark.parametrize("n", [2, -1])
def test_head(constructor_eager: Any, n: int) -> None:
    s = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]

    assert s.head(n).to_list() == [1, 2]
