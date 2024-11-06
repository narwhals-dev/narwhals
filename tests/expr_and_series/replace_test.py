from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_replace(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.select(nw.col("a").replace({1: 3, 2: 4}))
    assert_equal_data(result, {"a": [3, 4, 3]})


def test_replace_with_conflict(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.select(nw.col("a").replace({1: 3, 3: 4}))
    assert_equal_data(result, {"a": [3, 2, 4]})


def test_replace_series(constructor_eager: ConstructorEager) -> None:
    s = nw.from_native(constructor_eager({"a": [1, 2, 3]}))["a"]
    result = s.replace({1: 3, 3: 4})
    assert_equal_data({"a": result}, {"a": [3, 2, 4]})


def test_replace_different_dtype(constructor_eager: ConstructorEager) -> None:
    s = nw.from_native(constructor_eager({"a": [1, 2, 3]}))["a"]
    # Different libraries raise slightly different exceptions, but this is
    # probably fine, we just check that they all raise
    with pytest.raises(Exception):  # noqa: B017, PT011
        s.replace({1: "x", 3: "y", 2: "z"})
