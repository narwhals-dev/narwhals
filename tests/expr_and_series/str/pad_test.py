from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    uses_pyarrow_backend,
)


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


@pytest.mark.parametrize(
    ("data", "length", "fill_char", "expected"),
    [
        pytest.param(
            {"a": ["foo", "", None]},
            0,
            " ",
            {"start": ["foo", "", None], "end": ["foo", "", None]},
            id="length_zero",
        ),
        pytest.param(
            {"a": ["東京", "ab", None]},
            2,
            "日",
            {"start": ["東京", "ab", None], "end": ["東京", "ab", None]},
            id="unicode_exact",
        ),
    ],
)
def test_pad_noop_expr(
    constructor: Constructor,
    data: dict[str, list[str | None]],
    length: int,
    fill_char: str,
    expected: dict[str, list[str | None]],
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("a").str.pad_start(length, fill_char).alias("start"),
        nw.col("a").str.pad_end(length, fill_char).alias("end"),
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize("method", ["pad_start", "pad_end"])
@pytest.mark.parametrize("fill_char", ["ab", ""])
def test_pad_invalid_fill_char(
    constructor: Constructor, request: pytest.FixtureRequest, method: str, fill_char: str
) -> None:
    # Messages differ per backend, hence no `match`; dask surfaces it at collect time.
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="deferred error"))
    df = nw.from_native(constructor({"a": ["foo", None]}))
    with pytest.raises(ValueError):  # noqa: PT011
        df.select(getattr(nw.col("a").str, method)(5, fill_char))


@pytest.mark.parametrize("method", ["pad_start", "pad_end"])
def test_pad_negative_length_raises(
    constructor: Constructor, request: pytest.FixtureRequest, method: str
) -> None:
    # Old pandas is lenient here (except with a pyarrow-backed dtype).
    if (
        PANDAS_VERSION < (3,)
        and not uses_pyarrow_backend(constructor)
        and ("pandas" in str(constructor) or "dask" in str(constructor))
    ):
        request.applymarker(pytest.mark.xfail(reason="old pandas is lenient"))
    df = nw.from_native(constructor({"a": ["foo", None]}))
    expr = getattr(nw.col("a").str, method)(-1)
    # Broad `Exception`: error types differ per backend.
    if isinstance(df, nw.LazyFrame):
        with pytest.raises(Exception):  # noqa: B017, PT011
            df.select(expr).lazy().collect()
    else:
        with pytest.raises(Exception):  # noqa: B017, PT011
            df.select(expr)
