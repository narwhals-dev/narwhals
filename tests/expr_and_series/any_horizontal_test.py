from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


@pytest.mark.parametrize("expr1", ["a", nw.col("a")])
@pytest.mark.parametrize("expr2", ["b", nw.col("b")])
def test_anyh(constructor: Constructor, expr1: Any, expr2: Any) -> None:
    data = {"a": [False, False, True], "b": [False, True, True]}
    df = nw.from_native(constructor(data))
    result = df.select(any=nw.any_horizontal(expr1, expr2))

    expected = {"any": [False, True, True]}
    assert_equal_data(result, expected)


def test_anyh_kleene(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "cudf" in str(constructor):
        # https://github.com/rapidsai/cudf/issues/19171
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        # Dask infers `[True, None, None, None]` as `object` dtype, and then `__or__` fails.
        request.applymarker(pytest.mark.xfail)
    data = {"a": [True, True, False], "b": [True, None, None]}
    df = nw.from_native(constructor(data))
    result = df.select(any=nw.any_horizontal("a", "b"))
    if any(x in str(constructor) for x in ("pandas_constructor",)):
        expected: list[bool | None] = [True, True, False]
    else:
        expected = [True, True, None]
    assert_equal_data(result, {"any": expected})


def test_anyh_all(constructor: Constructor) -> None:
    data = {"a": [False, False, True], "b": [False, True, True]}
    df = nw.from_native(constructor(data))
    result = df.select(any=nw.any_horizontal(nw.all()))
    expected = {"any": [False, True, True]}
    assert_equal_data(result, expected)
    result = df.select(nw.any_horizontal(nw.all()))
    expected = {"a": [False, True, True]}
    assert_equal_data(result, expected)
