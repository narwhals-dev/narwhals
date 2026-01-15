from __future__ import annotations

import datetime as dt
import re
from typing import Any

import pytest

import narwhals as nw
from narwhals._utils import Implementation
from narwhals.exceptions import InvalidOperationError, NarwhalsError
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data


def test_concat_horizontal(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_left = nw.from_native(constructor_eager(data), eager_only=True)

    data_right = {"c": [6, 12, -1], "d": [0, -4, 2]}
    df_right = nw.from_native(constructor_eager(data_right), eager_only=True)

    result = nw.concat([df_left, df_right], how="horizontal")
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "c": [6, 12, -1],
        "d": [0, -4, 2],
    }
    assert_equal_data(result, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([])
    pattern = re.compile(r"horizontal.+not supported.+lazyframe", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nw.concat([df_left.lazy()], how="horizontal")


def test_concat_vertical(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_left = (
        nw.from_native(constructor(data)).lazy().rename({"a": "c", "b": "d"}).drop("z")
    )

    data_right = {"c": [6, 12, -1], "d": [0, -4, 2]}
    df_right = nw.from_native(constructor(data_right)).lazy()

    result = nw.concat([df_left, df_right], how="vertical")
    expected = {"c": [1, 3, 2, 6, 12, -1], "d": [4, 4, 6, 0, -4, 2]}
    assert_equal_data(result, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([], how="vertical")

    with pytest.raises(
        (Exception, TypeError),
        match=r"unable to vstack|inputs should all have the same schema",
    ):
        nw.concat([df_left, df_right.rename({"d": "i"})], how="vertical").collect()
    with pytest.raises(
        (Exception, TypeError),
        match=r"unable to vstack|unable to append|inputs should all have the same schema",
    ):
        nw.concat([df_left, df_left.select("d")], how="vertical").collect()


def test_concat_diagonal(constructor: Constructor) -> None:
    data_1 = {"a": [1, 3], "b": [4, 6]}
    data_2 = {"a": [100, 200], "z": ["x", "y"]}
    expected = {
        "a": [1, 3, 100, 200],
        "b": [4, 6, None, None],
        "z": [None, None, "x", "y"],
    }

    df_1 = nw.from_native(constructor(data_1)).lazy()
    df_2 = nw.from_native(constructor(data_2)).lazy()

    result = nw.concat([df_1, df_2], how="diagonal")

    assert_equal_data(result, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([], how="diagonal")


def test_concat_diagonal_invalid(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    data_1 = {"a": [1, 3], "b": [4, 6]}
    data_2 = {
        "a": [dt.datetime(2000, 1, 1), dt.datetime(2000, 1, 2)],
        "b": [4, 6],
        "z": ["x", "y"],
    }
    df_1 = nw.from_native(constructor(data_1)).lazy()
    bad_schema = nw.from_native(constructor(data_2)).lazy()
    impl = df_1.implementation
    request.applymarker(
        pytest.mark.xfail(
            impl not in {Implementation.IBIS, Implementation.POLARS},
            reason=f"{impl!r} does not validate schemas for `concat(how='diagonal')",
        )
    )
    expected_exception: Any = InvalidOperationError, TypeError
    if impl.is_polars() and POLARS_VERSION < (1,):  # pragma: no cover
        expected_exception = *expected_exception, NarwhalsError

    with pytest.raises(expected_exception, match=r"same schema"):
        nw.concat([df_1, bad_schema], how="diagonal").collect().to_dict(as_series=False)
