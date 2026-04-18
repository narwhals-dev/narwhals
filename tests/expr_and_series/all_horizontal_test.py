from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import Any

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data


def test_allh(nw_frame_constructor: Constructor) -> None:
    data = {"a": [False, False, True], "b": [False, True, True]}
    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(all=nw.all_horizontal("a", nw.col("b"), ignore_nulls=True))

    expected = {"all": [False, False, True]}
    assert_equal_data(result, expected)


def test_all_ignore_nulls(nw_frame_constructor: Constructor) -> None:
    if "dask" in str(nw_frame_constructor):
        # Dask infers `[True, None, None, None]` as `object` dtype, and then `__or__` fails.
        # test it below separately
        pytest.skip()
    data = {"a": [True, True, False], "b": [True, None, None]}
    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(any=nw.all_horizontal("a", "b", ignore_nulls=True))
    expected = [True, True, False]
    assert_equal_data(result, {"any": expected})


def test_allh_kleene(
    nw_frame_constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(nw_frame_constructor):
        # https://github.com/rapidsai/cudf/issues/19171
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(nw_frame_constructor):
        # Dask infers `[True, None, None, None]` as `object` dtype, and then `__or__` fails.
        # test it below separately
        pytest.skip()
    context = (
        pytest.raises(ValueError, match="ignore_nulls")
        if "pandas_constructor" in str(nw_frame_constructor)
        else does_not_raise()
    )
    data = {"a": [True, True, False], "b": [True, None, None]}
    df = nw.from_native(nw_frame_constructor(data))
    with context:
        result = df.select(all=nw.all_horizontal("a", "b", ignore_nulls=False))
        expected = [True, None, False]
        assert_equal_data(result, {"all": expected})


def test_anyh_dask(nw_frame_constructor: Constructor) -> None:
    if "dask" not in str(nw_frame_constructor):
        pytest.skip()
    import dask.dataframe as dd
    import pandas as pd

    data = {"a": [True, True, False], "b": [True, None, None]}
    df = nw.from_native(dd.from_pandas(pd.DataFrame(data, dtype="Boolean[pyarrow]")))
    result = df.select(all=nw.all_horizontal("a", "b", ignore_nulls=True))
    expected: list[bool | None] = [True, True, False]
    assert_equal_data(result, {"all": expected})
    result = df.select(all=nw.all_horizontal("a", "b", ignore_nulls=False))
    expected = [True, None, False]
    assert_equal_data(result, {"all": expected})

    # No nulls, NumPy-backed
    data = {"a": [True, True, False], "b": [True, False, False]}
    df = nw.from_native(dd.from_pandas(pd.DataFrame(data)))
    result = df.select(all=nw.all_horizontal("a", "b", ignore_nulls=True))
    expected = [True, False, False]
    assert_equal_data(result, {"all": expected})


def test_allh_series(nw_eager_constructor: ConstructorEager) -> None:
    data = {"a": [False, False, True], "b": [False, True, True]}
    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    result = df.select(all=nw.all_horizontal(df["a"], df["b"], ignore_nulls=True))

    expected = {"all": [False, False, True]}
    assert_equal_data(result, expected)


def test_allh_all(nw_frame_constructor: Constructor) -> None:
    data = {"a": [False, False, True], "b": [False, True, True]}
    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(all=nw.all_horizontal(nw.all(), ignore_nulls=True))
    expected = {"all": [False, False, True]}
    assert_equal_data(result, expected)
    result = df.select(nw.all_horizontal(nw.all(), ignore_nulls=True))
    expected = {"a": [False, False, True]}
    assert_equal_data(result, expected)


def test_allh_nth(nw_frame_constructor: Constructor) -> None:
    if "polars" in str(nw_frame_constructor) and POLARS_VERSION < (1, 0):
        pytest.skip()
    data = {"a": [False, False, True], "b": [False, True, True]}
    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(nw.all_horizontal(nw.nth(0, 1), ignore_nulls=True))
    expected = {"a": [False, False, True]}
    assert_equal_data(result, expected)
    result = df.select(nw.all_horizontal(nw.col("a"), nw.nth(0), ignore_nulls=True))
    expected = {"a": [False, False, True]}
    assert_equal_data(result, expected)


def test_allh_iterator(nw_frame_constructor: Constructor) -> None:
    def iter_eq(items: Any, /) -> Any:
        for column, value in items:
            yield nw.col(column) == value

    data = {"a": [1, 2, 3, 3, 3], "b": ["b", "b", "a", "a", "b"]}
    df = nw.from_native(nw_frame_constructor(data))
    expr_items = [("a", 3), ("b", "b")]
    expected = {"a": [3], "b": ["b"]}

    eager = nw.all_horizontal(list(iter_eq(expr_items)), ignore_nulls=True)
    assert_equal_data(df.filter(eager), expected)
    unpacked = nw.all_horizontal(*iter_eq(expr_items), ignore_nulls=True)
    assert_equal_data(df.filter(unpacked), expected)
    lazy = nw.all_horizontal(iter_eq(expr_items), ignore_nulls=True)

    assert_equal_data(df.filter(lazy), expected)
    assert_equal_data(df.filter(lazy), expected)
    assert_equal_data(df.filter(lazy), expected)


def test_horizontal_expressions_empty(nw_frame_constructor: Constructor) -> None:
    data = {"a": [False, False, True], "b": [False, True, True]}
    df = nw.from_native(nw_frame_constructor(data))
    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*all_horizontal"
    ):
        df.select(nw.all_horizontal(ignore_nulls=True))
    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*any_horizontal"
    ):
        df.select(nw.any_horizontal(ignore_nulls=True))
    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*mean_horizontal"
    ):
        df.select(nw.mean_horizontal())
    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*sum_horizontal"
    ):
        df.select(nw.sum_horizontal())

    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*max_horizontal"
    ):
        df.select(nw.max_horizontal())

    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*min_horizontal"
    ):
        df.select(nw.min_horizontal())
