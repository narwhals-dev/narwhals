from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


@pytest.mark.parametrize(("threshold", "expected"), [(0, False), (10, True)])
def test_is_empty(
    constructor_eager: ConstructorEager, threshold: Any, expected: Any
) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw, eager_only=True)
    result = df.filter(nw.col("a") > threshold).is_empty()
    assert result == expected
