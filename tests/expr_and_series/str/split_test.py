from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"s": ["foo bar", "foo_bar", "foo_bar_baz", "foo,bar"]}


@pytest.mark.parametrize(
    ("by", "inclusive", "expected"),
    [
        (
            "_",
            False,
            {"s": [["foo bar"], ["foo", "bar"], ["foo", "bar", "baz"], ["foo,bar"]]},
        ),
        (
            "_",
            True,
            {"s": [["foo bar"], ["foo_", "bar"], ["foo_", "bar_", "baz"], ["foo,bar"]]},
        ),
    ],
)
def test_str_split(
    constructor: Constructor,
    by: str,
    inclusive: bool,  # noqa: FBT001
    expected: Any,
) -> None:
    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("s").str.split(by=by, inclusive=inclusive))
    assert_equal_data(result_frame, expected)


@pytest.mark.parametrize(
    ("by", "inclusive", "expected"),
    [
        (
            "_",
            False,
            {"s": [["foo bar"], ["foo", "bar"], ["foo", "bar", "baz"], ["foo,bar"]]},
        ),
        (
            "_",
            True,
            {"s": [["foo bar"], ["foo_", "bar"], ["foo_", "bar_", "baz"], ["foo,bar"]]},
        ),
    ],
)
def test_str_split_series(
    constructor_eager: ConstructorEager,
    by: str | None,
    inclusive: bool,  # noqa: FBT001
    expected: Any,
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["s"].str.split(by=by, inclusive=inclusive)
    assert_equal_data(({"s": result_series}), expected.all())
