from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import PythonLiteral


def test_anyh(constructor: Constructor) -> None:
    data = {"a": [False, False, True], "b": [False, True, True]}
    df = nw.from_native(constructor(data))
    result = df.select(any=nw.any_horizontal(nw.col("a"), "b", ignore_nulls=True))

    expected = {"any": [False, True, True]}
    assert_equal_data(result, expected)


def test_anyh_kleene(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "cudf" in str(constructor):
        # https://github.com/rapidsai/cudf/issues/19171
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        # Dask infers `[True, None, None, None]` as `object` dtype, and then `__or__` fails.
        # test it below separately
        pytest.skip()
    context = (
        pytest.raises(ValueError, match="ignore_nulls")
        if "pandas_constructor" in str(constructor)
        else does_not_raise()
    )
    data = {"a": [True, True, False], "b": [True, None, None]}
    df = nw.from_native(constructor(data))
    with context:
        result = df.select(any=nw.any_horizontal("a", "b", ignore_nulls=False))
        expected = [True, True, None]
        assert_equal_data(result, {"any": expected})


def test_anyh_ignore_nulls(constructor: Constructor) -> None:
    if "dask" in str(constructor):
        # Dask infers `[True, None, None, None]` as `object` dtype, and then `__or__` fails.
        # test it below separately
        pytest.skip()
    data = {"a": [True, True, False], "b": [True, None, None]}
    df = nw.from_native(constructor(data))
    result = df.select(any=nw.any_horizontal("a", "b", ignore_nulls=True))
    expected = [True, True, False]
    assert_equal_data(result, {"any": expected})


def test_anyh_dask(constructor: Constructor) -> None:
    if "dask" not in str(constructor):
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


def test_anyh_all(constructor: Constructor) -> None:
    if "dask" in str(constructor):
        # Can't use `ignore_nulls` for NumPy-backed Dask, test it separately below
        pytest.skip()
    data = {"a": [False, False, True], "b": [False, True, True]}
    df = nw.from_native(constructor(data))
    result = df.select(any=nw.any_horizontal(nw.all(), ignore_nulls=False))
    expected = {"any": [False, True, True]}
    assert_equal_data(result, expected)
    result = df.select(nw.any_horizontal(nw.all(), ignore_nulls=False))
    expected = {"a": [False, True, True]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("exprs", "name"),
    [
        ((nw.col("a"), False), "a"),
        ((nw.col("a"), nw.lit(False)), "a"),
        ((False, nw.col("a")), "literal"),
        ((nw.lit(False), nw.col("a")), "literal"),
    ],
)
def test_anyh_with_scalars(
    constructor: Constructor, exprs: tuple[PythonLiteral | nw.Expr, ...], name: str
) -> None:
    data = {"a": [False, True]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.any_horizontal(*exprs, ignore_nulls=True))
    assert_equal_data(result, {name: [False, True]})
