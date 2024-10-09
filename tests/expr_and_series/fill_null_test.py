from contextlib import nullcontext as does_not_raise
from typing import Any

import pytest

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


def test_fill_null_exceptions(constructor: Constructor) -> None:
    data = {
        "a": [0.0, None, 2, 3, 4],
    }
    df = nw.from_native(constructor(data))

    with pytest.raises(ValueError, match="cannot specify both `value` and `strategy`"):
        df.with_columns(nw.col("a").fill_null(value=99, strategy="forward"))

    with pytest.raises(
        ValueError, match="must specify either a fill `value` or `strategy`"
    ):
        df.with_columns(nw.col("a").fill_null())

    with pytest.raises(ValueError, match="strategy not supported:"):
        df.with_columns(nw.col("a").fill_null(strategy="invalid"))  # type: ignore  # noqa: PGH003


def test_fill_null_strategies_with_limit_as_none(constructor: Constructor) -> None:
    data_limits = {
        "a": [1, None, None, None, 5, 6, None, None, None, 10],
        "b": ["a", None, None, None, "b", "c", None, None, None, "d"],
    }
    df = nw.from_native(constructor(data_limits))

    result_forward = df.with_columns(
        nw.col("a", "b").fill_null(strategy="forward", limit=None)
    )

    expected_forward = {
        "a": [1, 1, 1, 1, 5, 6, 6, 6, 6, 10],
        "b": ["a", "a", "a", "a", "b", "c", "c", "c", "c", "d"],
    }
    compare_dicts(result_forward, expected_forward)

    result_backward = df.with_columns(
        nw.col("a", "b").fill_null(strategy="backward", limit=None)
    )
    expected_backward = {
        "a": [1, 5, 5, 5, 5, 6, 10, 10, 10, 10],
        "b": ["a", "b", "b", "b", "b", "c", "d", "d", "d", "d"],
    }
    compare_dicts(result_backward, expected_backward)


def test_fill_null_limits(constructor: Constructor) -> None:
    context = (
        pytest.raises(
            NotImplementedError,
            match="The limit keyword is not supported",
        )
        if "cudf" in str(constructor)
        else does_not_raise()
    )
    data_limits = {
        "a": [1, None, None, None, 5, 6, None, None, None, 10],
        "b": ["a", None, None, None, "b", "c", None, None, None, "d"],
    }
    df = nw.from_native(constructor(data_limits))
    with context:
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


def test_fill_null_series(constructor_eager: Any) -> None:
    data_series_float = {
        "a": [0.0, 1, None, 2, None, 3],
    }
    df_float = nw.from_native(constructor_eager(data_series_float), eager_only=True)

    expected_float = {
        "a_zero_digit": [0.0, 1, 0, 2, 0, 3],
    }
    result_float = df_float.select(
        a_zero_digit=df_float["a"].fill_null(value=0),
    )

    compare_dicts(result_float, expected_float)

    data_series_str = {
        "a": ["a", None, "c", None, "e"],
    }
    df_str = nw.from_native(constructor_eager(data_series_str), eager_only=True)

    expected_str = {
        "a_z_str": ["a", "z", "c", "z", "e"],
    }

    result_str = df_str.select(
        a_z_str=df_str["a"].fill_null(value="z"),
    )

    compare_dicts(result_str, expected_str)


def test_fill_null_series_limits(constructor_eager: Any) -> None:
    context = (
        pytest.raises(
            NotImplementedError,
            match="The limit keyword is not supported",
        )
        if "cudf" in str(constructor_eager)
        else does_not_raise()
    )
    data_series_float = {
        "a": [0.0, 1, None, None, 2, None, None, 3],
        "b": ["", "a", None, None, "c", None, None, "e"],
    }
    df = nw.from_native(constructor_eager(data_series_float), eager_only=True)

    with context:
        expected_forward = {
            "a_forward": [0.0, 1, 1, float("nan"), 2, 2, float("nan"), 3],
            "b_forward": ["", "a", "a", None, "c", "c", None, "e"],
        }
        result_forward = df.select(
            a_forward=df["a"].fill_null(strategy="forward", limit=1),
            b_forward=df["b"].fill_null(strategy="forward", limit=1),
        )

        compare_dicts(result_forward, expected_forward)

        expected_backward = {
            "a_backward": [0.0, 1, float("nan"), 2, 2, float("nan"), 3, 3],
            "b_backward": ["", "a", None, "c", "c", None, "e", "e"],
        }

        result_backward = df.select(
            a_backward=df["a"].fill_null(strategy="backward", limit=1),
            b_backward=df["b"].fill_null(strategy="backward", limit=1),
        )

        compare_dicts(result_backward, expected_backward)


def test_fill_null_series_limit_as_none(constructor_eager: Any) -> None:
    data_series = {
        "a": [1, None, None, None, 5, 6, None, None, None, 10],
    }
    df = nw.from_native(constructor_eager(data_series), eager_only=True)

    expected_forward = {
        "a_forward": [1, 1, 1, 1, 5, 6, 6, 6, 6, 10],
        "a_backward": [1, 5, 5, 5, 5, 6, 10, 10, 10, 10],
    }
    result_forward = df.select(
        a_forward=df["a"].fill_null(strategy="forward", limit=None),
        a_backward=df["a"].fill_null(strategy="backward", limit=None),
    )

    compare_dicts(result_forward, expected_forward)

    data_series_str = {
        "a": ["a", None, None, None, "b", "c", None, None, None, "d"],
    }

    df_str = nw.from_native(constructor_eager(data_series_str), eager_only=True)

    expected_forward_str = {
        "a_forward": ["a", "a", "a", "a", "b", "c", "c", "c", "c", "d"],
        "a_backward": ["a", "b", "b", "b", "b", "c", "d", "d", "d", "d"],
    }

    result_forward_str = df_str.select(
        a_forward=df_str["a"].fill_null(strategy="forward", limit=None),
        a_backward=df_str["a"].fill_null(strategy="backward", limit=None),
    )
    compare_dicts(result_forward_str, expected_forward_str)


def test_fill_null_series_exceptions(constructor_eager: Any) -> None:
    data_series_float = {
        "a": [0.0, 1, None, 2, None, 3],
    }
    df_float = nw.from_native(constructor_eager(data_series_float), eager_only=True)

    with pytest.raises(ValueError, match="cannot specify both `value` and `strategy`"):
        df_float.select(
            a_zero_digit=df_float["a"].fill_null(value=0, strategy="forward"),
        )

    with pytest.raises(
        ValueError, match="must specify either a fill `value` or `strategy`"
    ):
        df_float.select(
            a_zero_digit=df_float["a"].fill_null(),
        )

    with pytest.raises(ValueError, match="strategy not supported:"):
        df_float.select(
            a_zero_digit=df_float["a"].fill_null(strategy="invalid"),  # type: ignore  # noqa: PGH003
        )
