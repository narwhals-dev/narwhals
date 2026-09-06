from __future__ import annotations

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


def test_pad_length_zero_expr(constructor: Constructor) -> None:
    # Length 0: every string is already at least that long, so input is
    # returned unchanged (per docstring).
    df = nw.from_native(constructor({"a": ["foo", "", None]}))
    result = df.select(
        nw.col("a").str.pad_start(0).alias("start"),
        nw.col("a").str.pad_end(0).alias("end"),
    )
    expected = {"start": ["foo", "", None], "end": ["foo", "", None]}
    assert_equal_data(result, expected)


def test_pad_unicode_exact_length_expr(constructor: Constructor) -> None:
    # Unicode input already exactly `length` characters long stays unchanged.
    df = nw.from_native(constructor({"a": ["東京", "ab", None]}))
    result = df.select(
        nw.col("a").str.pad_start(2, "日").alias("start"),
        nw.col("a").str.pad_end(2, "日").alias("end"),
    )
    expected = {"start": ["東京", "ab", None], "end": ["東京", "ab", None]}
    assert_equal_data(result, expected)
