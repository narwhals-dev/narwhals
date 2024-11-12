from __future__ import annotations

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_select(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.select("a")
    expected = {"a": [1, 3, 2]}
    assert_equal_data(result, expected)


def test_empty_select(constructor: Constructor) -> None:
    result = nw.from_native(constructor({"a": [1, 2, 3]})).lazy().select()
    assert result.collect().shape == (0, 0)


def test_non_string_select() -> None:
    df = nw.from_native(pd.DataFrame({0: [1, 2], "b": [3, 4]}))
    result = nw.to_native(df.select(nw.col(0)))  # type: ignore[arg-type]
    expected = pd.Series([1, 2], name=0).to_frame()
    pd.testing.assert_frame_equal(result, expected)


def test_non_string_select_invalid() -> None:
    df = nw.from_native(pd.DataFrame({0: [1, 2], "b": [3, 4]}))
    with pytest.raises(TypeError, match="\n\nHint: if you were trying to select"):
        nw.to_native(df.select(0))  # type: ignore[arg-type]


def test_select_boolean_cols() -> None:
    df = nw.from_native(pd.DataFrame({True: [1, 2], False: [3, 4]}))
    result = df.group_by(True).agg(nw.col(False).max())  # type: ignore[arg-type]# noqa: FBT003
    assert_equal_data(result, {True: [1, 2]})  # type: ignore[dict-item]
    result = df.select(nw.col([False, True]))  # type: ignore[list-item]
    assert_equal_data(result, {True: [1, 2], False: [3, 4]})  # type: ignore[dict-item]
