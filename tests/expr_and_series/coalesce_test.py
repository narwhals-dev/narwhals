from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


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
        implicit_lit=nw.coalesce("a", "b", "c", -1),
    )
    expected = {
        "no_lit": [0, None, 3, 5, 3],
        "explicit_lit": [0, -1, 3, 5, 3],
        "implicit_lit": [0, -1, 3, 5, 3],
    }
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
