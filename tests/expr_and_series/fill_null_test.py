from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts


def test_fill_null(constructor: Constructor) -> None:
    data = {
        "a": [0.0, None, 2, 3, 4],
        "b": [1.0, None, None, 5, 3],
        "c": [5.0, None, 3, 2, 1],
    }
    df = nw.from_native(constructor(data))

    result = df.with_columns(nw.col("a", "b", "c").fill_null(value=99))
    expected = {
        "a": [0.0, 99, 2, 3, 4],
        "b": [1.0, 99, 99, 5, 3],
        "c": [5.0, 99, 3, 2, 1],
    }
    compare_dicts(result, expected)


def test_fill_null_strategies(constructor: Constructor) -> None:
    data_strategies = {"a": [1, 2, 3], "b": [4, None, 5]}
    df = nw.from_native(constructor(data_strategies))

    result = df.with_columns(nw.col("a", "b").fill_null(strategy="forward"))
    expected = {
        "a": [1, 2, 3],
        "b": [4, 4, 5],
    }
    compare_dicts(result, expected)

    result = df.with_columns(nw.col("a", "b").fill_null(strategy="backward"))
    expected = {
        "a": [1, 2, 3],
        "b": [4, 5, 5],
    }
    compare_dicts(result, expected)


def test_fill_null_limits(constructor: Constructor) -> None:
    data_limits = {
        "a": [1, None, None, None, 5, 6, None, None, None, 10],
        "b": ["a", None, None, None, "b", "c", None, None, None, "d"],
    }
    df = nw.from_native(constructor(data_limits))

    result_forward = df.with_columns(
        nw.col("a", "b").fill_null(strategy="forward", limit=2)
    )
    expected_forward = {
        "a": [1, 1, 1, float("nan"), 5, 6, 6, 6, float("nan"), 10],
        "b": ["a", "a", "a", None, "b", "c", "c", "c", None, "d"],
    }
    compare_dicts(result_forward, expected_forward)

    result_backward = df.with_columns(
        nw.col("a", "b").fill_null(strategy="backward", limit=2)
    )

    expected_backward = {
        "a": [1, float("nan"), 5, 5, 5, 6, float("nan"), 10, 10, 10],
        "b": ["a", None, "b", "b", "b", "c", None, "d", "d", "d"],
    }
    compare_dicts(result_backward, expected_backward)

    result_none_limit = df.with_columns(
        nw.col("a", "b").fill_null(strategy="backward", limit=None)
    )

    expected_none_limit = {
        "a": [1, 5, 5, 5, 5, 6, 10, 10, 10, 10],
        "b": ["a", "b", "b", "b", "b", "c", "d", "d", "d", "d"],
    }
    compare_dicts(result_none_limit, expected_none_limit)


def test_fill_null_series(constructor_eager: Any) -> None:
    data_series_float = {
        "a": [0.0, 1, None, 2, None, 3],
    }
    df_float = nw.from_native(constructor_eager(data_series_float), eager_only=True)

    expected_float = {
        "a_zero_digit": [0.0, 1, 0, 2, 0, 3],
        "a_forward_strategy": [0.0, 1, 1, 2, 2, 3],
        "a_backward_strategy": [0.0, 1, 2, 2, 3, 3],
    }
    result_float = df_float.select(
        a_zero_digit=df_float["a"].fill_null(value=0),
        a_forward_strategy=df_float["a"].fill_null(strategy="forward"),
        a_backward_strategy=df_float["a"].fill_null(strategy="backward"),
    )

    compare_dicts(result_float, expected_float)

    data_series_str = {
        "a": ["a", None, "c", None, "e"],
    }
    df_str = nw.from_native(constructor_eager(data_series_str), eager_only=True)

    expected_str = {
        "a_forward_strategy": ["a", "a", "c", "c", "e"],
        "a_backward_strategy": ["a", "c", "c", "e", "e"],
    }

    result_str = df_str.select(
        a_forward_strategy=df_str["a"].fill_null(strategy="forward"),
        a_backward_strategy=df_str["a"].fill_null(strategy="backward"),
    )

    compare_dicts(result_str, expected_str)
