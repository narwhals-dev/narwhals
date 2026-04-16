from __future__ import annotations

import narwhals as nw
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
