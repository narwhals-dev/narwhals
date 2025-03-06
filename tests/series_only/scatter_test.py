from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_scatter(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "modin" in str(constructor_eager):
        # https://github.com/modin-project/modin/issues/7392
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [142, 124, 132]}), eager_only=True
    )
    result = df.with_columns(
        df["a"].scatter([0, 1], [999, 888]),
        df["b"].scatter([0, 2, 1], df["b"]),
    )
    expected = {
        "a": [999, 888, 3],
        "b": [142, 132, 124],
    }
    assert_equal_data(result, expected)


def test_scatter_unchanged(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [142, 124, 132]}), eager_only=True
    )
    df.with_columns(
        df["a"].scatter([0, 1], [999, 888]), df["b"].scatter([0, 2, 1], [142, 124, 132])
    )
    expected = {
        "a": [1, 2, 3],
        "b": [142, 124, 132],
    }
    assert_equal_data(df, expected)


def test_single_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [142, 124, 132]}), eager_only=True
    )
    s = df["a"]
    s.scatter([0, 1], [999, 888])
    expected = {"a": [1, 2, 3]}
    assert_equal_data({"a": s}, expected)


def test_scatter_unordered_indices(constructor_eager: ConstructorEager) -> None:
    data = {
        "a": [16, 12, 10, 9, 6, 5, 2],
    }
    indices = [6, 1, 0, 5, 3, 2, 4]
    df = nw.from_native(constructor_eager(data))
    result = df["a"].scatter(indices, df["a"])
    assert_equal_data({"a": result}, {"a": [10, 12, 5, 6, 2, 9, 16]})
