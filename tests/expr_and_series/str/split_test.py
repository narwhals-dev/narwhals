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
def test_str_split(nw_frame_constructor: Constructor, by: str, expected: Any) -> None:
    if "cudf" not in str(nw_frame_constructor) and (
        str(nw_frame_constructor).startswith("pandas")
        and "pyarrow" not in str(nw_frame_constructor)
    ):
        df = nw.from_native(nw_frame_constructor(data))
        msg = re.escape("This operation requires a pyarrow-backed series. ")
        with pytest.raises(TypeError, match=msg):
            df.select(nw.col("s").str.split(by=by))
        return
    df = nw.from_native(nw_frame_constructor(data))
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
    nw_eager_constructor: ConstructorEager, by: str, expected: Any
) -> None:
    if "cudf" not in str(nw_eager_constructor) and (
        str(nw_eager_constructor).startswith("pandas")
        and "pyarrow" not in str(nw_eager_constructor)
    ):
        df = nw.from_native(nw_eager_constructor(data), eager_only=True)
        msg = re.escape("This operation requires a pyarrow-backed series. ")
        with pytest.raises(TypeError, match=msg):
            df["s"].str.split(by=by)
        return
    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    result_series = df["s"].str.split(by=by)
    assert_equal_data({"s": result_series}, expected)
