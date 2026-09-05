from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import PYARROW_VERSION, ConstructorEager, assert_equal_data

# Includes a null to check it is excluded from the returned categories.
data = {"a": ["one", "two", "two", None]}
expected = {"a": ["one", "two"]}


def test_get_categories_eager(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (15, 0, 0):
        pytest.skip()

    if "polars" in str(constructor_eager):
        reason = "https://github.com/narwhals-dev/narwhals/issues/3097"
        request.applymarker(pytest.mark.xfail(reason=reason, strict=False))

    result = nw.from_native(constructor_eager(data), eager_only=True).select(
        nw.col("a").cast(nw.Categorical).cat.get_categories()
    )
    assert_equal_data(result, expected)


def test_get_categories_series(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (15, 0, 0):
        pytest.skip()

    if "polars" in str(constructor_eager):
        reason = "https://github.com/narwhals-dev/narwhals/issues/3097"
        request.applymarker(pytest.mark.xfail(reason=reason, strict=False))

    result = (
        nw.from_native(constructor_eager(data), eager_only=True)["a"]
        .cast(nw.Categorical)
        .cat.get_categories()
    )
    assert_equal_data({"a": result}, expected)


def test_get_categories_lazy(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (15, 0, 0):
        pytest.skip()

    if "polars" in str(constructor_eager):
        reason = "https://github.com/narwhals-dev/narwhals/issues/3097"
        request.applymarker(pytest.mark.xfail(reason=reason, strict=False))

    df = nw.from_native(constructor_eager(data)).lazy()
    expr = nw.col("a").cast(nw.Categorical).cat.get_categories()
    msg = "Length-changing expressions are not supported for use in LazyFrame"
    with pytest.raises(InvalidOperationError, match=msg):
        df.select(expr).collect()

    result = df.select(expr.min())
    expected = {"a": ["one"]}
    assert_equal_data(result, expected)
