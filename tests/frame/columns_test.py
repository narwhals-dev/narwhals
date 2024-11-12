from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from tests.utils import Constructor


@pytest.mark.filterwarnings("ignore:Determining|Resolving.*")
def test_columns(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.columns
    expected = ["a", "b", "z"]
    assert result == expected
