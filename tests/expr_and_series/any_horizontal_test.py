from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


def test_anyh(nw_frame_constructor: Constructor) -> None:
    data = {"a": [False, False, True], "b": [False, True, True]}
    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(any=nw.any_horizontal(nw.col("a"), "b", ignore_nulls=True))

    expected = {"any": [False, True, True]}
    assert_equal_data(result, expected)


def test_anyh_kleene(
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
        result = df.select(any=nw.any_horizontal("a", "b", ignore_nulls=False))
        expected = [True, True, None]
        assert_equal_data(result, {"any": expected})


def test_anyh_ignore_nulls(nw_frame_constructor: Constructor) -> None:
    if "dask" in str(nw_frame_constructor):
        # Dask infers `[True, None, None, None]` as `object` dtype, and then `__or__` fails.
        # test it below separately
        pytest.skip()
    data = {"a": [True, True, False], "b": [True, None, None]}
    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(any=nw.any_horizontal("a", "b", ignore_nulls=True))
    expected = [True, True, False]
    assert_equal_data(result, {"any": expected})


def test_anyh_dask(nw_frame_constructor: Constructor) -> None:
    if "dask" not in str(nw_frame_constructor):
        pytest.skip()
    import dask.dataframe as dd
    import pandas as pd

    data = {"a": [True, True, False], "b": [True, None, None]}
    df = nw.from_native(dd.from_pandas(pd.DataFrame(data, dtype="Boolean[pyarrow]")))
    result = df.select(any=nw.any_horizontal("a", "b", ignore_nulls=True))
    expected: list[bool | None] = [True, True, False]
    assert_equal_data(result, {"any": expected})
    result = df.select(any=nw.any_horizontal("a", "b", ignore_nulls=False))
    expected = [True, True, None]
    assert_equal_data(result, {"any": expected})

    # No nulls, NumPy-backed
    data = {"a": [True, True, False], "b": [True, False, False]}
    df = nw.from_native(dd.from_pandas(pd.DataFrame(data)))
    result = df.select(any=nw.any_horizontal("a", "b", ignore_nulls=True))
    expected = [True, True, False]
    assert_equal_data(result, {"any": expected})


def test_anyh_all(nw_frame_constructor: Constructor) -> None:
    if "dask" in str(nw_frame_constructor):
        # Can't use `ignore_nulls` for NumPy-backed Dask, test it separately below
        pytest.skip()
    data = {"a": [False, False, True], "b": [False, True, True]}
    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(any=nw.any_horizontal(nw.all(), ignore_nulls=False))
    expected = {"any": [False, True, True]}
    assert_equal_data(result, expected)
    result = df.select(nw.any_horizontal(nw.all(), ignore_nulls=False))
    expected = {"a": [False, True, True]}
    assert_equal_data(result, expected)
