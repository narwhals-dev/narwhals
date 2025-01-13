from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_concat_horizontal(
    constructor: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df_left = nw.from_native(constructor(data)).lazy()

    data_right = {"c": [6, 12, -1], "d": [0, -4, 2]}
    df_right = nw.from_native(constructor(data_right)).lazy()

    result = nw.concat([df_left, df_right], how="horizontal")
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8, 9],
        "c": [6, 12, -1],
        "d": [0, -4, 2],
    }
    assert_equal_data(result, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([])


def test_concat_vertical(constructor: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
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
        match="unable to vstack|inputs should all have the same schema",
    ):
        nw.concat([df_left, df_right.rename({"d": "i"})], how="vertical").collect()
    with pytest.raises(
        (Exception, TypeError),
        match="unable to vstack|unable to append|inputs should all have the same schema",
    ):
        nw.concat([df_left, df_left.select("d")], how="vertical").collect()


def test_concat_diagonal(
    constructor: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
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
