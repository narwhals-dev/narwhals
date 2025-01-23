from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.exceptions import ColumnNotFoundError
from narwhals.exceptions import InvalidIntoExprError
from tests.utils import DASK_VERSION
from tests.utils import PANDAS_VERSION
from tests.utils import POLARS_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data


class Foo: ...


def test_select(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    result = df.select("a")
    expected = {"a": [1, 3, 2]}
    assert_equal_data(result, expected)


def test_empty_select(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    result = nw.from_native(constructor({"a": [1, 2, 3]})).lazy().select()
    assert result.collect().shape == (0, 0)


def test_non_string_select() -> None:
    df = nw.from_native(pd.DataFrame({0: [1, 2], "b": [3, 4]}))
    result = nw.to_native(df.select(nw.col(0)))  # type: ignore[arg-type]
    expected = pd.Series([1, 2], name=0).to_frame()
    pd.testing.assert_frame_equal(result, expected)


def test_int_select_pandas() -> None:
    df = nw.from_native(pd.DataFrame({0: [1, 2], "b": [3, 4]}))
    with pytest.raises(InvalidIntoExprError, match="\n\nHint:\n- if you were trying"):
        nw.to_native(df.select(0))  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_select", [None, 0, Foo()])
def test_invalid_select(
    constructor: Constructor, invalid_select: Any, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and not isinstance(invalid_select, Foo):
        # https://github.com/narwhals-dev/narwhals/issues/1390
        request.applymarker(pytest.mark.xfail)
    with pytest.raises(InvalidIntoExprError):
        nw.from_native(constructor({"a": [1, 2, 3]})).select(invalid_select)


def test_select_boolean_cols(request: pytest.FixtureRequest) -> None:
    if PANDAS_VERSION < (1, 1):
        # bug in old pandas
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(pd.DataFrame({True: [1, 2], False: [3, 4]}), eager_only=True)
    result = df.group_by(True).agg(nw.col(False).max())  # type: ignore[arg-type]# noqa: FBT003
    assert_equal_data(result.to_dict(as_series=False), {True: [1, 2]})  # type: ignore[dict-item]
    result = df.select(nw.col([False, True]))  # type: ignore[list-item]
    assert_equal_data(result.to_dict(as_series=False), {True: [1, 2], False: [3, 4]})  # type: ignore[dict-item]


def test_comparison_with_list_error_message() -> None:
    msg = "Expected scalar value, Series, or Expr, got list of : <class 'int'>"
    with pytest.raises(ValueError, match=msg):
        nw.from_native(pa.chunked_array([[1, 2, 3]]), series_only=True) == [1, 2, 3]  # noqa: B015
    with pytest.raises(ValueError, match=msg):
        nw.from_native(pd.Series([[1, 2, 3]]), series_only=True) == [1, 2, 3]  # noqa: B015


def test_missing_columns(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    selected_columns = ["a", "e", "f"]
    msg = r"\n\nHint: Did you mean one of these columns: \['a', 'b', 'z'\]?"
    if "polars" in str(constructor):
        # In the lazy case, Polars only errors when we call `collect`,
        # and we have no way to recover exactly which columns the user
        # tried selecting. So, we just emit their message (which varies
        # across versions...)
        msg = "e|f"
        if isinstance(df, nw.LazyFrame):
            with pytest.raises(ColumnNotFoundError, match=msg):
                df.select(selected_columns).collect()
        else:
            with pytest.raises(ColumnNotFoundError, match=msg):
                df.select(selected_columns)
        if POLARS_VERSION >= (1,):
            # Old Polars versions wouldn't raise an error
            # at all here
            if isinstance(df, nw.LazyFrame):
                with pytest.raises(ColumnNotFoundError, match=msg):
                    df.drop(selected_columns, strict=True).collect()
            else:
                with pytest.raises(ColumnNotFoundError, match=msg):
                    df.drop(selected_columns, strict=True)
        else:  # pragma: no cover
            pass
    else:
        with pytest.raises(ColumnNotFoundError, match=msg):
            df.select(selected_columns)
        with pytest.raises(ColumnNotFoundError, match=msg):
            df.drop(selected_columns, strict=True)
        with pytest.raises(ColumnNotFoundError, match=msg):
            df.select(nw.col("fdfa"))


def test_left_to_right_broadcasting(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor) and DASK_VERSION < (2024, 10):
        request.applymarker(pytest.mark.xfail)
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 1, 2], "b": [4, 5, 6]}))
    result = df.select(nw.col("a") + nw.col("b").sum())
    expected = {"a": [16, 16, 17]}
    assert_equal_data(result, expected)
    result = df.select(nw.col("b").sum() + nw.col("a"))
    expected = {"b": [16, 16, 17]}
    assert_equal_data(result, expected)
    result = df.select(nw.col("b").sum() + nw.col("a").sum())
    expected = {"b": [19]}
    assert_equal_data(result, expected)
