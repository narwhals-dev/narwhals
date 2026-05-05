from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.testing.constructors import get_backend_constructor


def test_from_native() -> None:
    ibis_constructor = get_backend_constructor("ibis")
    if not ibis_constructor.is_available:
        pytest.skip()
    df = ibis_constructor({"a": [1, 2, 3], "b": [4, 5, 6]}, nw)
    assert df.columns == ["a", "b"]
