from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import _1DArray

data: dict[str, Any] = {
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
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    with pytest.raises(
        NotImplementedError, match="Slicing with step is not supported on PyArrow tables"
    ):
        nw.from_native(pa.table(data))[1::2]
    with pytest.raises(
        NotImplementedError, match="Slicing with step is not supported on PyArrow tables"
    ):
        nw.from_native(pa.chunked_array([data["a"]]), series_only=True)[1::2]


def test_slice_lazy_fails() -> None:
    pytest.importorskip("polars")
    import polars as pl

    with pytest.raises(TypeError, match="Slicing is not supported on LazyFrame"):
        _ = nw.from_native(pl.LazyFrame(data))[1:]


def test_slice_int(constructor_eager: ConstructorEager) -> None:
    result = nw.from_native(constructor_eager(data), eager_only=True)[1]
    assert_equal_data(result, {"a": [2], "b": [12]})


def test_slice_fails(constructor_eager: ConstructorEager) -> None:
    class Foo: ...

    with pytest.raises(TypeError, match="Unexpected type.*, got:"):
        nw.from_native(constructor_eager(data), eager_only=True)[Foo()]  # type: ignore[call-overload, unused-ignore]


def test_gather(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df[[0, 3, 1]]
    expected = {"a": [1.0, 4.0, 2.0], "b": [11, 14, 12]}
    assert_equal_data(result, expected)
    arr = cast("_1DArray", np.array([0, 3, 1]))
    result = df[arr]
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

    result: Any = {"b": df[[0, 3, 1], 1]}
    assert_equal_data(result, expected)
    arr = cast("_1DArray", np.array([0, 3, 1]))
    result = {"b": df[arr, "b"]}
    assert_equal_data(result, expected)


def test_slice_both_list_of_ints(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df[[0, 1], [0, 2]]
    expected = {"a": [1, 2], "c": [7, 8]}
    assert_equal_data(result, expected)


def test_slice_both_tuple(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor_eager):
        # https://github.com/rapidsai/cudf/issues/18556
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df[(0, 1), ("a", "c")]
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


def test_slice_item(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2], "b": [4, 5]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    assert df[0, 0] == 1


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
    ("row_idx", "col"), [([0, 2], slice(1)), ((0, 2), slice(1)), (range(2), slice(1))]
)
def test_get_item_works_with_tuple_and_list_and_range_row_indexing_and_slice_col_indexing(
    constructor_eager: ConstructorEager,
    row_idx: list[int] | tuple[int] | range,
    col: slice,
) -> None:
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    nw_df[row_idx, col]


@pytest.mark.parametrize(
    ("row_idx", "col"), [([0, 2], "a"), ((0, 2), "a"), (range(2), "a")]
)
def test_get_item_works_with_tuple_and_list_indexing_and_str(
    constructor_eager: ConstructorEager, row_idx: list[int] | tuple[int] | range, col: str
) -> None:
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    nw_df[row_idx, col]


def test_getitem_ndarray_columns(constructor_eager: ConstructorEager) -> None:
    data = {"col1": ["a", "b", "c", "d"], "col2": np.arange(4), "col3": [4, 3, 2, 1]}
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    arr = np.arange(2)
    result = nw_df[:, arr]
    expected = {"col1": ["a", "b", "c", "d"], "col2": [0, 1, 2, 3]}
    assert_equal_data(result, expected)


def test_getitem_ndarray_columns_labels(constructor_eager: ConstructorEager) -> None:
    data = {"col1": ["a", "b", "c", "d"], "col2": np.arange(4), "col3": [4, 3, 2, 1]}
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    arr: np.ndarray[tuple[int], np.dtype[Any]] = np.array(["col1", "col2"])  # pyright: ignore[reportAssignmentType]
    result = nw_df[:, arr]
    expected = {"col1": ["a", "b", "c", "d"], "col2": [0, 1, 2, 3]}
    assert_equal_data(result, expected)


def test_getitem_negative_slice(constructor_eager: ConstructorEager) -> None:
    data = {"col1": ["a", "b", "c", "d"], "col2": np.arange(4), "col3": [4, 3, 2, 1]}
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    result = nw_df[-3:-2, ["col3", "col1"]]
    expected = {"col3": [3], "col1": ["b"]}
    assert_equal_data(result, expected)
    result = nw_df[-3:-2]
    expected = {"col1": ["b"], "col2": [1], "col3": [3]}
    assert_equal_data(result, expected)
    result_s = nw_df["col1"][-3:-2]
    expected = {"col1": ["b"]}
    assert_equal_data({"col1": result_s}, expected)


def test_zeroth_row_no_columns(constructor_eager: ConstructorEager) -> None:
    data = {"col1": ["a", "b", "c", "d"], "col2": np.arange(4), "col3": [4, 3, 2, 1]}
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    columns: list[str] = []
    result = nw_df[0, columns]
    assert result.shape == (0, 0)


def test_single_tuple(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 3]}
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    # Technically works but we should probably discourage it
    # OK if overloads don't match it.
    result = nw_df[[0, 1],]  # type: ignore[index]
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)


def test_triple_tuple(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 3]}
    with pytest.raises(TypeError, match="Tuples cannot"):
        nw.from_native(constructor_eager(data), eager_only=True)[(1, 2, 3)]


def test_slice_with_series(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "pandas_pyarrow" in str(constructor_eager):
        # https://github.com/pandas-dev/pandas/issues/61311
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 2, 3], "c": [0, 2, 1]}
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    result = nw_df[nw_df["c"]]
    expected = {"a": [1, 3, 2], "c": [0, 1, 2]}
    assert_equal_data(result, expected)
    result = nw_df[nw_df["c"], ["a"]]
    expected = {"a": [1, 3, 2]}
    assert_equal_data(result, expected)


def test_horizontal_slice_with_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2], "c": [0, 2], "d": ["c", "a"]}
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    result = nw_df[nw_df["d"]]
    expected = {"c": [0, 2], "a": [1, 2]}
    assert_equal_data(result, expected)


def test_horizontal_slice_with_series_2(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "pandas_pyarrow" in str(constructor_eager):
        # https://github.com/pandas-dev/pandas/issues/61311
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 2], "c": [0, 2], "d": ["c", "a"]}
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    result = nw_df[:, nw_df["c"]]
    expected = {"a": [1, 2], "d": ["c", "a"]}
    assert_equal_data(result, expected)


def test_native_slice_series(constructor_eager: ConstructorEager) -> None:
    s = nw.from_native(constructor_eager({"a": [0, 2, 1]}), eager_only=True)["a"]
    result = {"a": s[s.to_native()]}
    expected = {"a": [0, 1, 2]}
    assert_equal_data(result, expected)


def test_pandas_non_str_columns() -> None:
    # The general rule with getitem is: ints are always treated as positions. The rest, we should
    # be able to hand down to the native frame. Here we check what happens for pandas with
    # datetime column names.
    df = nw.from_native(
        pd.DataFrame({datetime(2020, 1, 1): [1, 2, 3], datetime(2020, 1, 2): [4, 5, 6]}),
        eager_only=True,
    )
    result = df[:, [datetime(2020, 1, 1)]]  # type: ignore[index]
    expected = {datetime(2020, 1, 1): [1, 2, 3]}
    assert result.to_dict(as_series=False) == expected


def test_select_rows_by_name(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [0, 2, 1]}), eager_only=True)
    with pytest.raises(TypeError, match="Unexpected type"):
        df["a", :]  # type: ignore[index]
