from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import DuplicateError
from tests.utils import Constructor, assert_equal_data


def test_rename(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    result = df.rename({"a": "foo-bar", "b": "foo bar", "z": "z.b"})
    expected = {"foo-bar": [1, 3, 2], "foo bar": [4, 4, 6], "z.b": [7.0, 8.0, 9.0]}
    assert_equal_data(result, expected)
    result = df.rename({"b": "z", "z": "y"})
    expected = {"a": [1, 3, 2], "z": [4, 4, 6], "y": [7.0, 8.0, 9.0]}
    assert_equal_data(result, expected)


def test_rename_swap(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6]}
    df = nw.from_native(constructor(data))
    result = df.rename({"a": "b", "b": "a"})
    expected = {"b": [1, 3, 2], "a": [4, 4, 6]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("mapping", [{"b": "a"}, {"a": "x", "b": "x"}])
def test_rename_duplicate_raises(
    constructor: Constructor, mapping: dict[str, str]
) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]})).lazy()
    with pytest.raises(DuplicateError):
        df.rename(mapping).collect()


@pytest.mark.parametrize("mapping", [{"b": "a"}, {"a": "x", "b": "x"}])
def test_rename_duplicate_raises_with_followup_op(
    constructor: Constructor, mapping: dict[str, str]
) -> None:
    # DuckDB used to silently rename the duplicate to `a_1` once another op followed
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]})).lazy()
    with pytest.raises(DuplicateError):
        df.rename(mapping).sort(*mapping.values()).collect()
