from __future__ import annotations

import re
from typing import Any

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"s": ["foo bar", "foo_bar", "foo_bar_baz", "foo,bar"]}


@pytest.mark.parametrize(
    ("by", "expected"),
    [
        ("_", {"s": [["foo bar"], ["foo", "bar"], ["foo", "bar", "baz"], ["foo,bar"]]}),
        (",", {"s": [["foo bar"], ["foo_bar"], ["foo_bar_baz"], ["foo", "bar"]]}),
    ],
)
def test_str_split(constructor: Constructor, by: str, expected: Any) -> None:
    if "cudf" not in str(constructor) and (
        constructor.__name__.startswith("pandas")
        and "pyarrow" not in constructor.__name__
    ):
        df = nw.from_native(constructor(data))
        msg = re.escape("This operation requires a pyarrow-backed series. ")
        with pytest.raises(TypeError, match=msg):
            df.select(nw.col("s").str.split(by=by))
        return
    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("s").str.split(by=by))
    assert_equal_data(result_frame, expected)


@pytest.mark.parametrize(
    ("by", "expected"),
    [
        ("_", {"s": [["foo bar"], ["foo", "bar"], ["foo", "bar", "baz"], ["foo,bar"]]}),
        (",", {"s": [["foo bar"], ["foo_bar"], ["foo_bar_baz"], ["foo", "bar"]]}),
    ],
)
def test_str_split_series(
    constructor_eager: ConstructorEager, by: str, expected: Any
) -> None:
    if "cudf" not in str(constructor_eager) and (
        constructor_eager.__name__.startswith("pandas")
        and "pyarrow" not in constructor_eager.__name__
    ):
        df = nw.from_native(constructor_eager(data), eager_only=True)
        msg = re.escape("This operation requires a pyarrow-backed series. ")
        with pytest.raises(TypeError, match=msg):
            df["s"].str.split(by=by)
        return
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result_series = df["s"].str.split(by=by)
    assert_equal_data({"s": result_series}, expected)
