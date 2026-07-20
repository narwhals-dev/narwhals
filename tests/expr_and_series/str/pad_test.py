from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


def test_str_pad_start_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager({"a": ["foo", "longer_foo", "longest_fooooooo", "hi", None]})
    )

    result = {
        "padded": df["a"].str.pad_start(10),
        "padded_len": df["a"].str.pad_start(10).str.len_chars(),
    }
    expected = {
        "padded": ["       foo", "longer_foo", "longest_fooooooo", "        hi", None],
        "padded_len": [10, 10, 16, 10, None],
    }

    assert_equal_data(result, expected)


def test_str_pad_start_expr(constructor: Constructor) -> None:
    df = nw.from_native(
        constructor({"a": ["foo", "longer_foo", "longest_fooooooo", "hi", None]})
    )

    result = df.select(
        nw.col("a").str.pad_start(10).alias("padded"),
        nw.col("a").str.pad_start(10).str.len_chars().alias("padded_len"),
    )
    expected = {
        "padded": ["       foo", "longer_foo", "longest_fooooooo", "        hi", None],
        "padded_len": [10, 10, 16, 10, None],
    }

    assert_equal_data(result, expected)


def test_str_pad_end_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager({"a": ["foo", "longer_foo", "longest_fooooooo", "hi", None]})
    )

    result = {
        "padded": df["a"].str.pad_end(10),
        "padded_len": df["a"].str.pad_end(10).str.len_chars(),
    }
    expected = {
        "padded": ["foo       ", "longer_foo", "longest_fooooooo", "hi        ", None],
        "padded_len": [10, 10, 16, 10, None],
    }

    assert_equal_data(result, expected)


def test_str_pad_end_expr(constructor: Constructor) -> None:
    df = nw.from_native(
        constructor({"a": ["foo", "longer_foo", "longest_fooooooo", "hi", None]})
    )

    result = df.select(
        nw.col("a").str.pad_end(10).alias("padded"),
        nw.col("a").str.pad_end(10).str.len_chars().alias("padded_len"),
    )
    expected = {
        "padded": ["foo       ", "longer_foo", "longest_fooooooo", "hi        ", None],
        "padded_len": [10, 10, 16, 10, None],
    }

    assert_equal_data(result, expected)


def test_pad_start_unicode_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": ["Café", "345", "東京", None]}))

    result = df.select(nw.col("a").str.pad_start(6, "日"))
    expected = {"a": ["日日Café", "日日日345", "日日日日東京", None]}

    assert_equal_data(result, expected)


def test_pad_start_unicode_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": ["Café", "345", "東京", None]}))

    result = {"a": df["a"].str.pad_start(6, "日")}
    expected = {"a": ["日日Café", "日日日345", "日日日日東京", None]}

    assert_equal_data(result, expected)


def test_pad_end_unicode_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": ["Café", "345", "東京", None]}))

    result = df.select(nw.col("a").str.pad_end(6, "日"))
    expected = {"a": ["Café日日", "345日日日", "東京日日日日", None]}

    assert_equal_data(result, expected)


def test_pad_end_unicode_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": ["Café", "345", "東京", None]}))

    result = {"a": df["a"].str.pad_end(6, "日")}
    expected = {"a": ["Café日日", "345日日日", "東京日日日日", None]}

    assert_equal_data(result, expected)


def test_str_pad_negative_length_raises(constructor_eager: ConstructorEager) -> None:
    # Same divergence as zfill, since the length reaches the same backend calls. See
    # the note in zfill_test.py for why the message is pinned and not just the type.
    s = nw.from_native(constructor_eager({"a": ["abc"]}), eager_only=True)["a"]
    msg = r"`length` must be non-negative but got -1"
    with pytest.raises(nw.exceptions.InvalidOperationError, match=msg):
        s.str.pad_start(-1)
    with pytest.raises(nw.exceptions.InvalidOperationError, match=msg):
        s.str.pad_end(-1)


def test_str_pad_negative_length_expr_raises(constructor: Constructor) -> None:
    # Separate from the series test so the lazy backends are covered too.
    df = nw.from_native(constructor({"a": ["abc"]}))
    msg = r"`length` must be non-negative but got -1"
    with pytest.raises(nw.exceptions.InvalidOperationError, match=msg):
        df.select(nw.col("a").str.pad_start(-1))
    with pytest.raises(nw.exceptions.InvalidOperationError, match=msg):
        df.select(nw.col("a").str.pad_end(-1))
