from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}


@pytest.mark.filterwarnings("ignore:Determining|Resolving.*")
def test_columns(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    result = df.columns
    expected = ["a", "b", "z"]
    assert result == expected


def test_iter_columns(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    expected = df.to_dict(as_series=True)
    result = {series.name: series for series in df.iter_columns()}
    assert result == expected
