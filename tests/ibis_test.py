from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.testing.constructors import ConstructorName


def test_from_native() -> None:
    if not (name := ConstructorName.IBIS).is_available:
        pytest.skip()
    df = nw.from_native(name.constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert df.columns == ["a", "b"]
