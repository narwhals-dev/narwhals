from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor

# A reduction's result dtype should not depend on whether any value survives.
NUMERIC_REDUCTIONS = ["mean", "min", "max", "std", "var", "median", "sum"]


@pytest.mark.parametrize("reduction", NUMERIC_REDUCTIONS)
def test_numeric_reduction_keeps_dtype_when_all_null(
    constructor: Constructor, request: pytest.FixtureRequest, reduction: str
) -> None:
    if reduction != "sum" and any(
        x in str(constructor)
        for x in ("pandas_nullable", "pandas_pyarrow", "modin_pyarrow")
    ):
        # `sum` is the exception: it reduces an all-null column to a typed zero.
        reason = "an all-null reduction yields `pd.NA`, which carries no dtype"
        request.applymarker(pytest.mark.xfail(reason=reason))

    df = nw.from_native(constructor({"a": [1.0, 2.0]}))
    all_null = df.with_columns(a=nw.when(nw.col("a") < 0).then(nw.col("a")))
    expr = getattr(nw.col("a"), reduction)()
    assert (
        all_null.select(expr).lazy().collect_schema()["a"]
        == df.select(expr).lazy().collect_schema()["a"]
    )


def test_string_reduction_keeps_dtype_when_all_null(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("pandas_constructor", "modin_constructor")):
        reason = "a reduction over an all-null object column returns `Float64` there"
        request.applymarker(pytest.mark.xfail(reason=reason))

    df = nw.from_native(constructor({"a": ["x", "y"]}))
    all_null = df.with_columns(a=nw.when(nw.col("a") == "zzz").then(nw.col("a")))
    assert all_null.select(nw.col("a").max()).lazy().collect_schema()["a"] == nw.String


def test_all_null_reduction_supports_str_namespace(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(
        x in str(constructor) for x in ("pandas_constructor", "modin_constructor", "dask")
    ):
        reason = "an all-null reduction loses the string dtype on pandas-like backends"
        request.applymarker(pytest.mark.xfail(reason=reason))

    df = nw.from_native(constructor({"a": ["x", "y"]}))
    all_null = df.with_columns(a=nw.when(nw.col("a") == "zzz").then(nw.col("a")))
    result = all_null.select(nw.col("a").max().str.to_uppercase())
    assert_equal_data(result, {"a": [None]})
