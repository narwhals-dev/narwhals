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


@pytest.mark.parametrize("method", ["pad_start", "pad_end"])
@pytest.mark.parametrize("fill_char", ["ab", ""])
def test_pad_invalid_fill_char(
    constructor: Constructor, request: pytest.FixtureRequest, method: str, fill_char: str
) -> None:
    # Padding requires a single character, matching polars (`ValueError`;
    # pyarrow's `ArrowInvalid` subclasses it). Dask only surfaces it at
    # collect time instead.
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="deferred error"))
    df = nw.from_native(constructor({"a": ["foo", None]}))
    # No `match`: polars, pyarrow, and narwhals-raised messages each word the
    # single-character requirement differently.
    with pytest.raises(ValueError):  # noqa: PT011
        df.select(getattr(nw.col("a").str, method)(5, fill_char))


@pytest.mark.parametrize("method", ["pad_start", "pad_end"])
def test_pad_negative_length_raises(constructor: Constructor, method: str) -> None:
    # A negative length raises on every backend (each with its own error type)
    # instead of returning the input unchanged.
    df = nw.from_native(constructor({"a": ["foo", None]}))
    expr = getattr(nw.col("a").str, method)(-1)
    # Broad `Exception`: every backend raises, but each with its own error
    # type. The pinned contract is only "raises rather than padding".
    if isinstance(df, nw.LazyFrame):
        with pytest.raises(Exception):  # noqa: B017, PT011
            df.select(expr).lazy().collect()
    else:
        with pytest.raises(Exception):  # noqa: B017, PT011
            df.select(expr)
