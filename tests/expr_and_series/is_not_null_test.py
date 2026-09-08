from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


def test_not_null(constructor: Constructor) -> None:
    data_na = {"a": [None, 3, 2], "z": [7.0, None, None]}
    expected = {"a": [False, True, True], "z": [True, False, False]}
    df = nw.from_native(constructor(data_na))
    result = df.select(nw.col("a").is_not_null(), nw.col("z").is_not_null())

    assert_equal_data(result, expected)


def test_not_null_series(constructor_eager: ConstructorEager) -> None:
    data_na = {"a": [None, 3, 2], "z": [7.0, None, None]}
    expected = {"a": [False, True, True], "z": [True, False, False]}
    df = nw.from_native(constructor_eager(data_na), eager_only=True)
    result = {"a": df["a"].is_not_null(), "z": df["z"].is_not_null()}

    assert_equal_data(result, expected)


def test_not_null_is_inverse_of_null(constructor: Constructor) -> None:
    # `is_not_null` should always be the exact negation of `is_null`.
    data_na = {"a": [None, 3, 2], "z": [7.0, None, None]}
    df = nw.from_native(constructor(data_na))
    result = df.select(
        (nw.col("a").is_not_null() == ~nw.col("a").is_null()).alias("a"),
        (nw.col("z").is_not_null() == ~nw.col("z").is_null()).alias("z"),
    )

    assert_equal_data(result, {"a": [True, True, True], "z": [True, True, True]})
