from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": ["fdas", "edfas"]}

short_data = {"a": ["fdas", "ab", ""]}


@pytest.mark.parametrize(
    ("offset", "length", "expected"),
    [
        (1, 2, {"a": ["da", "df"]}),
        (-2, None, {"a": ["as", "as"]}),
        # length=0 must slice to an empty string, not be treated as "to the end".
        (1, 0, {"a": ["", ""]}),
    ],
)
def test_str_slice(
    constructor: Constructor,
    offset: int,
    length: int | None,
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("a").str.slice(offset, length))
    assert_equal_data(result_frame, expected)


@pytest.mark.parametrize(
    ("offset", "length", "expected"),
    [
        (1, 2, {"a": ["da", "df"]}),
        (-2, None, {"a": ["as", "as"]}),
        # length=0 must slice to an empty string, not be treated as "to the end".
        (1, 0, {"a": ["", ""]}),
    ],
)
def test_str_slice_series(
    constructor_eager: ConstructorEager,
    offset: int,
    length: int | None,
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.slice(offset, length)
    assert_equal_data({"a": result_series}, expected)


@pytest.mark.parametrize(
    ("offset", "length", "expected"),
    [
        (-3, None, {"a": ["das", "ab", ""]}),
        (-3, 3, {"a": ["das", "ab", ""]}),
        (-3, 10, {"a": ["das", "ab", ""]}),
        # When `offset` reaches before the start, Polars shortens `length` by the overhang.
        (-3, 2, {"a": ["da", "a", ""]}),
        (-3, 1, {"a": ["d", "", ""]}),
        (-5, 2, {"a": ["f", "", ""]}),
        (-10, 3, {"a": ["", "", ""]}),
    ],
)
def test_str_slice_negative_offset(
    constructor: Constructor,
    offset: int,
    length: int | None,
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor(short_data))
    result_frame = df.select(nw.col("a").str.slice(offset, length))
    assert_equal_data(result_frame, expected)


def test_str_slice_negative_length_raises(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(ValueError, match="non-negative"):
        df.select(nw.col("a").str.slice(1, -1))
    with pytest.raises(ValueError, match="non-negative"):
        df["a"].str.slice(1, -1)
