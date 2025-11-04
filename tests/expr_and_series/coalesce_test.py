from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


def test_coalesce_numeric(constructor: Constructor) -> None:
    data = {
        "a": [0, None, None, None, None],
        "b": [1, None, None, 5, 3],
        "c": [5, None, 3, 2, 1],
    }
    df = nw.from_native(constructor(data))

    result = df.select(
        no_lit=nw.coalesce("a", "b", "c"),
        explicit_lit=nw.coalesce("a", "b", "c", nw.lit(-1)),
    )
    expected = {"no_lit": [0, None, 3, 5, 3], "explicit_lit": [0, -1, 3, 5, 3]}
    assert_equal_data(result, expected)


def test_coalesce_strings(constructor: Constructor) -> None:
    data = {
        "a": ["0", None, None, None, None],
        "b": ["1", None, None, "5", "3"],
        "c": ["5", None, "3", "2", "1"],
    }
    df = nw.from_native(constructor(data))

    result = df.select(
        no_lit=nw.coalesce("a", "b", "c"),
        explicit_lit=nw.coalesce("a", "b", "c", nw.lit("xyz")),
    )
    expected = {
        "no_lit": ["0", None, "3", "5", "3"],
        "explicit_lit": ["0", "xyz", "3", "5", "3"],
    }
    assert_equal_data(result, expected)


def test_coalesce_series(constructor_eager: ConstructorEager) -> None:
    data = {
        "a": ["0", None, None, None, None],
        "b": ["1", None, None, "5", "3"],
        "c": ["5", None, "3", "2", "1"],
    }
    df = nw.from_native(constructor_eager(data))

    result = df.select(result=nw.coalesce("a", "b", df["c"]))
    expected = {"result": ["0", None, "3", "5", "3"]}
    assert_equal_data(result, expected)


def test_coalesce_raises_non_expr(constructor: Constructor) -> None:
    data = {
        "a": ["0", None, None, None, None],
        "b": ["1", None, None, "5", "3"],
        "c": ["5", None, "3", "2", "1"],
    }
    df = nw.from_native(constructor(data))

    with pytest.raises(TypeError, match="All arguments to `coalesce` must be of type"):
        df.select(implicit_lit=nw.coalesce("a", "b", "c", 10))


def test_coalesce_multi_output(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/3277
    if "dask" in str(constructor):
        # Dask infers the type to be `string` :shrug:
        # https://github.com/narwhals-dev/narwhals/issues/3279
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(
        constructor(
            {
                "col1": [True, None, False, False, None],
                "col2": [True, False, True, False, None],
                "col3": [True, True, None, False, True],
            }
        )
    )
    result = df.select(nw.coalesce(nw.all()))
    expected = {"col1": [True, False, False, False, True]}
    assert_equal_data(result, expected)
