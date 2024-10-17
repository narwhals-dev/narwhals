from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import assert_equal_data

data = {"a": ["one", "two", "two"]}


def test_get_categories(request: pytest.FixtureRequest, constructor_eager: Any) -> None:
    if "pyarrow_table" in str(constructor_eager) and parse_version(
        pa.__version__
    ) < parse_version("15.0.0"):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    df = df.select(nw.col("a").cast(nw.Categorical))
    expected = {"a": ["one", "two"]}

    result_expr = df.select(nw.col("a").cat.get_categories())
    assert_equal_data(result_expr, expected)

    result_series = df["a"].cat.get_categories()
    assert_equal_data({"a": result_series}, expected)


def test_get_categories_pyarrow() -> None:
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
