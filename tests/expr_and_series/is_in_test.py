from __future__ import annotations

import re

import pytest

import narwhals as nw
from tests.utils import (
    Constructor,
    ConstructorEager,
    IntoIterable,
    assert_equal_data,
    assert_equal_series,
)

data = {"a": [1, 4, 2, 5]}


def test_expr_is_in(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").is_in([4, 5]))
    expected = {"a": [False, True, False, True]}

    assert_equal_data(result, expected)


def test_expr_is_in_empty_list(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").is_in([]))
    expected = {"a": [False, False, False, False]}

    assert_equal_data(result, expected)


def test_expr_is_in_iterable(constructor: Constructor, into_iter_4: IntoIterable) -> None:
    df = nw.from_native(constructor(data))
    expected = {"a": [False, True, True, False]}
    iterable = into_iter_4((4, 2))
    expr = nw.col("a").is_in(iterable)
    result = df.select(expr)
    assert_equal_data(result, expected)
    # NOTE: For an `Iterator`, this will fail if we haven't collected it first
    repeated = df.select(expr)
    assert_equal_data(repeated, expected)


def test_ser_is_in(constructor_eager: ConstructorEager) -> None:
    ser = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = {"a": ser.is_in([4, 5])}
    expected = {"a": [False, True, False, True]}

    assert_equal_data(result, expected)


def test_is_in_other(constructor: Constructor) -> None:
    df_raw = constructor(data)
    msg = re.escape(
        "Narwhals `is_in` doesn't accept expressions as an argument, as opposed to "
        "Polars. You should provide an iterable instead."
    )
    with pytest.raises(NotImplementedError, match=msg):
        nw.from_native(df_raw).with_columns(contains=nw.col("a").is_in("sets"))


def test_filter_is_in_with_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 4, 2, 5], "b": [1, 0, 2, 0]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.filter(nw.col("a").is_in(df["b"]))
    expected = {"a": [1, 2], "b": [1, 2]}
    assert_equal_data(result, expected)


def test_ser_is_in_iterable(
    constructor_eager: ConstructorEager,
    into_iter_16: IntoIterable,
    request: pytest.FixtureRequest,
) -> None:
    test_name = request.node.name
    # NOTE: This *could* be supported by using `ExtensionArray.tolist` (same path as numpy)
    request.applymarker(
        pytest.mark.xfail(
            ("polars" in test_name and "pandas" in test_name and "array" in test_name),
            raises=TypeError,
            reason="Polars doesn't support `pd.array`.\nhttps://github.com/pola-rs/polars/issues/22757",
        )
    )
    iterable = into_iter_16((4, 2))
    ser = nw.from_native(constructor_eager(data)).get_column("a")
    result = ser.is_in(iterable)
    expected = [False, True, True, False]
    assert_equal_series(result, expected, "a")
