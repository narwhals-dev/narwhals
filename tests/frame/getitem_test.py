from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {
    "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "b": [11, 12, 13, 14, 15, 16],
}


def test_slice_column(constructor_eager: ConstructorEager) -> None:
    result = nw.from_native(constructor_eager(data))["a"]
    assert isinstance(result, nw.Series)
    assert_equal_data({"a": result}, {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})


def test_slice_rows(constructor_eager: ConstructorEager) -> None:
    result = nw.from_native(constructor_eager(data))[1:]
    assert_equal_data(result, {"a": [2.0, 3.0, 4.0, 5.0, 6.0], "b": [12, 13, 14, 15, 16]})

    result = nw.from_native(constructor_eager(data))[2:4]
    assert_equal_data(result, {"a": [3.0, 4.0], "b": [13, 14]})


def test_slice_rows_with_step(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    result = nw.from_native(constructor_eager(data))[1::2]
    assert_equal_data(result, {"a": [2.0, 4.0, 6.0], "b": [12, 14, 16]})


def test_slice_rows_with_step_pyarrow() -> None:
    with pytest.raises(
        NotImplementedError,
        match="Slicing with step is not supported on PyArrow tables",
    ):
        nw.from_native(pa.table(data))[1::2]


def test_slice_lazy_fails() -> None:
    with pytest.raises(TypeError, match="Slicing is not supported on LazyFrame"):
        _ = nw.from_native(pl.LazyFrame(data))[1:]


def test_slice_int(constructor_eager: ConstructorEager) -> None:
    result = nw.from_native(constructor_eager(data), eager_only=True)[1]  # type: ignore[call-overload]
    assert_equal_data(result, {"a": [2], "b": [12]})


def test_slice_fails(constructor_eager: ConstructorEager) -> None:
    class Foo: ...

    with pytest.raises(TypeError, match="Expected str or slice, got:"):
        nw.from_native(constructor_eager(data), eager_only=True)[Foo()]  # type: ignore[call-overload]


def test_gather(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df[[0, 3, 1]]
    expected = {
        "a": [1.0, 4.0, 2.0],
        "b": [11, 14, 12],
    }
    assert_equal_data(result, expected)
    result = df[np.array([0, 3, 1])]
    assert_equal_data(result, expected)


def test_gather_pandas_index() -> None:
    # check that we're slicing positionally, and not on the pandas index
    df = pd.DataFrame({"a": [4, 1, 2], "b": [1, 4, 2]}, index=[2, 1, 3])
    result = nw.from_native(df, eager_only=True)[[1, 2]]
    expected = {"a": [1, 2], "b": [4, 2]}
    assert_equal_data(result, expected)

    result = nw.from_native(df, eager_only=True)[[1, 2], "a"].to_frame()
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)


def test_gather_rows_cols(constructor_eager: ConstructorEager) -> None:
    native_df = constructor_eager(data)
    df = nw.from_native(native_df, eager_only=True)

    expected = {"b": [11, 14, 12]}

    result = {"b": df[[0, 3, 1], 1]}
    assert_equal_data(result, expected)

    result = {"b": df[np.array([0, 3, 1]), "b"]}
    assert_equal_data(result, expected)


def test_slice_both_tuples_of_ints(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df[[0, 1], [0, 2]]
    expected = {"a": [1, 2], "c": [7, 8]}
    assert_equal_data(result, expected)


def test_slice_int_rows_str_columns(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df[[0, 1], ["a", "c"]]
    expected = {"a": [1, 2], "c": [7, 8]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("row_selector", "col_selector", "expected"),
    [
        ([0, 1], slice("b", "c"), {"b": [4, 5], "c": [7, 8]}),
        ([0, 1], slice(None, "c"), {"a": [1, 2], "b": [4, 5], "c": [7, 8]}),
        ([0, 1], slice("a", "d", 2), {"a": [1, 2], "c": [7, 8]}),
        ([0, 1], slice("b", None), {"b": [4, 5], "c": [7, 8], "d": [1, 4]}),
        ([0, 1], slice(1, 3), {"b": [4, 5], "c": [7, 8]}),
        ([0, 1], slice(None, 3), {"a": [1, 2], "b": [4, 5], "c": [7, 8]}),
        ([0, 1], slice(0, 4, 2), {"a": [1, 2], "c": [7, 8]}),
        ([0, 1], slice(1, None), {"b": [4, 5], "c": [7, 8], "d": [1, 4]}),
        (slice(None), ["b", "d"], {"b": [4, 5, 6], "d": [1, 4, 2]}),
        (slice(None), [0, 2], {"a": [1, 2, 3], "c": [7, 8, 9]}),
        (slice(None, 2), [0, 2], {"a": [1, 2], "c": [7, 8]}),
        (slice(None, 2), ["a", "c"], {"a": [1, 2], "c": [7, 8]}),
        (slice(1, None), [0, 2], {"a": [2, 3], "c": [8, 9]}),
        (slice(1, None), ["a", "c"], {"a": [2, 3], "c": [8, 9]}),
        (["b", "c"], None, {"b": [4, 5, 6], "c": [7, 8, 9]}),
        (slice(None, 2), None, {"a": [1, 2], "b": [4, 5], "c": [7, 8], "d": [1, 4]}),
        (slice(2, None), None, {"a": [3], "b": [6], "c": [9], "d": [2]}),
        (slice("a", "b"), None, {"a": [1, 2, 3], "b": [4, 5, 6]}),
        ((0, 1), slice(None), {"a": [1, 2], "b": [4, 5], "c": [7, 8], "d": [1, 4]}),
        ([0, 1], slice(None), {"a": [1, 2], "b": [4, 5], "c": [7, 8], "d": [1, 4]}),
        (
            [0, 1],
            ["a", "b", "c", "d"],
            {"a": [1, 2], "b": [4, 5], "c": [7, 8], "d": [1, 4]},
        ),
    ],
)
def test_slice_slice_columns(
    constructor_eager: ConstructorEager,
    row_selector: Any,
    col_selector: Any,
    expected: dict[str, list[Any]],
) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [1, 4, 2]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df[row_selector] if col_selector is None else df[row_selector, col_selector]
    assert_equal_data(result, expected)


def test_slice_invalid(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2], "b": [4, 5]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(TypeError, match="Hint:"):
        df[0, 0]


def test_slice_edge_cases(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [1, 4, 2]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    assert df[[], :].shape == (0, 4)
    assert df[:, []].shape == (0, 0)
    assert df[[]].shape == (0, 4)
    assert df[[], ["a"]].shape == (0, 1)
    assert df[:, :].shape == (3, 4)
    assert df[[], []].shape == (0, 0)


@pytest.mark.parametrize(
    ("row_idx", "col_idx"),
    [
        ([0, 2], [0]),
        ((0, 2), [0]),
        ([0, 2], (0,)),
        ((0, 2), (0,)),
        ([0, 2], range(1)),
        (range(2), [0]),
        (range(2), range(1)),
    ],
)
def test_get_item_works_with_tuple_and_list_and_range_row_and_col_indexing(
    constructor_eager: ConstructorEager,
    row_idx: list[int] | tuple[int] | range,
    col_idx: list[int] | tuple[int] | range,
) -> None:
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    nw_df[row_idx, col_idx]


@pytest.mark.parametrize(
    ("row_idx", "col"),
    [
        ([0, 2], slice(1)),
        ((0, 2), slice(1)),
        (range(2), slice(1)),
    ],
)
def test_get_item_works_with_tuple_and_list_and_range_row_indexing_and_slice_col_indexing(
    constructor_eager: ConstructorEager,
    row_idx: list[int] | tuple[int] | range,
    col: slice,
) -> None:
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    nw_df[row_idx, col]


@pytest.mark.parametrize(
    ("row_idx", "col"),
    [
        ([0, 2], "a"),
        ((0, 2), "a"),
        (range(2), "a"),
    ],
)
def test_get_item_works_with_tuple_and_list_indexing_and_str(
    constructor_eager: ConstructorEager,
    row_idx: list[int] | tuple[int] | range,
    col: str,
) -> None:
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    nw_df[row_idx, col]
