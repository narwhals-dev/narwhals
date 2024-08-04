from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_concat_horizontal(constructor: Any, request: Any) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df_left = nw.from_native(constructor(data))

    data_right = {"c": [6, 12, -1], "d": [0, -4, 2]}
    df_right = nw.from_native(constructor(data_right))

    result = nw.concat([df_left, df_right], how="horizontal")
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8, 9],
        "c": [6, 12, -1],
        "d": [0, -4, 2],
    }
    compare_dicts(result, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([])


def test_concat_vertical(constructor: Any, request: Any) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df_left = (
        nw.from_native(constructor(data)).rename({"a": "c", "b": "d"}).drop("z").lazy()
    )

    data_right = {"c": [6, 12, -1], "d": [0, -4, 2]}
    df_right = nw.from_native(constructor(data_right)).lazy()

    result = nw.concat([df_left, df_right], how="vertical")
    expected = {"c": [1, 3, 2, 6, 12, -1], "d": [4, 4, 6, 0, -4, 2]}
    compare_dicts(result, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([], how="vertical")

    with pytest.raises((Exception, TypeError), match="unable to vstack"):
        nw.concat([df_left, df_right.rename({"d": "i"})], how="vertical").collect()
