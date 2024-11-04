from __future__ import annotations

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"a": [1, 2, 3], "b": [4, 5, 6], "z": [7.0, 8.0, 9.0]}


def test_map_batches_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    expected = df.select(nw.col("a", "b").map_batches(lambda s: s + 1))
    assert_equal_data(expected, {"a": [2, 3, 4], "b": [5, 6, 7]})


def test_map_batches_expr_numpy(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    expected = df.select(
        nw.col("a")
        .map_batches(lambda s: s.to_numpy() + 1, return_dtype=nw.Float64())
        .sum()
    )
    assert_equal_data(expected, {"a": [9.0]})


def test_map_batches_expr_names(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    expected = nw.from_native(df.select(nw.all().map_batches(lambda x: x.to_numpy())))
    assert_equal_data(expected, {"a": [1, 2, 3], "b": [4, 5, 6], "z": [7.0, 8.0, 9.0]})


def test_map_batches_raise() -> None:
    pytest.importorskip("dask")
    pytest.importorskip("dask_expr", exc_type=ImportError)
    import dask.dataframe as dd

    df = nw.from_native(dd.from_pandas(pd.DataFrame({"a": [1, 2, 3]})))
    with pytest.raises(
        NotImplementedError, match="`Expr.map_batches` is not implemented for Dask yet"
    ):
        df.select(nw.col("a").map_batches(lambda x: x.to_numpy()))
