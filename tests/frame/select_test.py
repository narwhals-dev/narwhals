from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals._exceptions import ColumnNotFoundError
from tests.utils import PANDAS_VERSION
from tests.utils import POLARS_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_select(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.select("a")
    expected = {"a": [1, 3, 2]}
    assert_equal_data(result, expected)


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


def test_missing_columns(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    selected_columns = ["a", "e", "f"]
    msg = (
        r"The following columns were not found: \[.*\]"
        r"\n\nHint: Did you mean one of these columns: \['a', 'b', 'z'\]?"
    )
    if "polars" in str(constructor):
        # In the lazy case, Polars only errors when we call `collect`,
        # and we have no way to recover exactly which columns the user
        # tried selecting. So, we just emit their message (which varies
        # across versions...)
        msg = r"e"
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
