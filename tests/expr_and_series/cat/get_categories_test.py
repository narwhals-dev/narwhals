from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import PYARROW_VERSION, ConstructorEager, assert_equal_data

# Includes a null to check it is excluded from the returned categories.
data = {"a": ["one", "two", "two", None]}
expected = {"a": ["one", "two"]}

DEPRECATION_MSG = r"`(Expr|Series)\.cat\.get_categories` is deprecated"
# Modin's `astype(str)` returns a NumPy unicode dtype (e.g. `<U3`) where pandas
# returns `object`, and Narwhals maps that to `Unknown` rather than `String`.
MODIN_STR_CAST_REASON = "Modin returns a NumPy unicode dtype when casting to String"


def test_get_categories_eager(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (15, 0, 0):
        pytest.skip()

    if "modin" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail(reason=MODIN_STR_CAST_REASON))

    with pytest.warns(DeprecationWarning, match=DEPRECATION_MSG):
        expr = nw.col("a").cast(nw.Categorical).cat.get_categories()

    # `Expr.unique` doesn't guarantee any ordering, hence the `sort`.
    result = (
        nw.from_native(constructor_eager(data), eager_only=True).select(expr).sort("a")
    )
    assert_equal_data(result, expected)
    assert result.collect_schema() == {"a": nw.String}


def test_get_categories_series(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (15, 0, 0):
        pytest.skip()

    if "modin" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail(reason=MODIN_STR_CAST_REASON))

    s = nw.from_native(constructor_eager(data), eager_only=True)["a"].cast(nw.Categorical)
    with pytest.warns(DeprecationWarning, match=DEPRECATION_MSG):
        result = s.cat.get_categories()
    assert_equal_data({"a": result}, expected)
    assert result.dtype == nw.String


def test_get_categories_enum(constructor_eager: ConstructorEager) -> None:
    """Only the values which are *present* are returned, unused variants are dropped."""
    if "pyarrow_table" in str(constructor_eager):
        pytest.skip(reason="pyarrow doesn't support casting to Enum")

    dtype = nw.Enum(["one", "two", "three"])
    with pytest.warns(DeprecationWarning, match=DEPRECATION_MSG):
        expr = nw.col("a").cast(dtype).cat.get_categories()

    result = (
        nw.from_native(constructor_eager(data), eager_only=True).select(expr).sort("a")
    )
    assert_equal_data(result, expected)


def test_get_categories_lazy(constructor_eager: ConstructorEager) -> None:
    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (15, 0, 0):
        pytest.skip()

    df = nw.from_native(constructor_eager(data)).lazy()
    with pytest.warns(DeprecationWarning, match=DEPRECATION_MSG):
        expr = nw.col("a").cast(nw.Categorical).cat.get_categories()
    msg = "Length-changing expressions are not supported for use in LazyFrame"
    with pytest.raises(InvalidOperationError, match=msg):
        df.select(expr).collect()

    assert_equal_data(df.select(expr.min()), {"a": ["one"]})
