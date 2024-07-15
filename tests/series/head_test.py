from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw


@pytest.mark.parametrize("n", [2, -1])
def test_head(constructor_series: Any, n: int) -> None:
    s = nw.from_native(constructor_series([1, 2, 3]), series_only=True)

    assert s.head(n).to_list() == [1, 2]
