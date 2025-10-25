from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import PYARROW_VERSION, ConstructorEager, assert_equal_data

data = {"a": ["one", "two", "two"]}


def test_get_categories_eager(constructor_eager: ConstructorEager) -> None:
    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (15, 0, 0):
        pytest.skip()

    df = nw.from_native(constructor_eager(data), eager_only=True)
    df = df.select(nw.col("a").cast(nw.Categorical))
    expected = {"a": ["one", "two"]}

    result_expr = df.select(nw.col("a").cat.get_categories())
    assert_equal_data(result_expr, expected)

    result_series = df["a"].cat.get_categories()
    assert_equal_data({"a": result_series}, expected)


def test_get_categories_lazy(constructor_eager: ConstructorEager) -> None:
    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (15, 0, 0):
        pytest.skip()

    df = nw.from_native(constructor_eager(data)).lazy()
    expr = nw.col("a").cast(nw.Categorical).cat.get_categories()
    msg = "Length-changing expressions are not supported for use in LazyFrame"
    with pytest.raises(InvalidOperationError, match=msg):
        df.select(expr).collect()

    result = df.select(expr.min())
    expected = {"a": ["one"]}
    assert_equal_data(result, expected)


def test_get_categories_pyarrow() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    # temporary test until we have `cast` in pyarrow - later, fuse
    # this with test above
    table = pa.table(
        {"a": pa.array(["a", "b", None, "d"], pa.dictionary(pa.int64(), pa.utf8()))}
    )
    df = nw.from_native(table, eager_only=True)
    expected = {"a": ["a", "b", "d"]}

    result_expr = df.select(nw.col("a").cat.get_categories())
    assert_equal_data(result_expr, expected)

    result_series = df["a"].cat.get_categories()
    assert_equal_data({"a": result_series}, expected)
