from __future__ import annotations

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts


def test_select(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.select("a")
    expected = {"a": [1, 3, 2]}
    compare_dicts(result, expected)


def test_empty_select(constructor: Constructor) -> None:
    result = nw.from_native(constructor({"a": [1, 2, 3]})).lazy().select()
    assert result.collect().shape == (0, 0)


def test_non_string_select() -> None:
    df = nw.from_native(pd.DataFrame({0: [1, 2], "b": [3, 4]}))
    result = nw.to_native(df.select(nw.col(0)))  # type: ignore[arg-type]
    expected = pd.Series([1, 2], name=0).to_frame()
    pd.testing.assert_frame_equal(result, expected)


def test_non_string_select_invalid() -> None:
    df = nw.from_native(pd.DataFrame({0: [1, 2], "b": [3, 4]}))
    with pytest.raises(TypeError, match="\n\nHint: if you were trying to select"):
        nw.to_native(df.select(0))  # type: ignore[arg-type]


def test_dask_select_reduction_and_modify_index() -> None:
    pytest.importorskip("dask")
    pytest.importorskip("dask_expr", exc_type=ImportError)
    import dask.dataframe as dd

    data = {"a": [1, 3, 2], "b": [4, 4.0, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(dd.from_dict(data, npartitions=1))

    result = df.select(
        nw.col("a").head(2).sum(),
        nw.col("b").tail(2).mean(),
        nw.col("z").head(2),
    )
    expected = {"a": [4, 4], "b": [5, 5], "z": [7.0, 8]}
    compare_dicts(result, expected)

    # all reductions
    result = df.select(
        nw.col("a").head(2).sum(),
        nw.col("b").tail(2).mean(),
        nw.col("z").max(),
    )
    expected = {"a": [4], "b": [5], "z": [9]}
    compare_dicts(result, expected)
