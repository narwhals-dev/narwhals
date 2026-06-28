from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

data = {"a": [1, 2], "b": [3, 4]}


def test_get_column(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.get_column("a")
    assert_equal_data({"a": result}, {"a": [1, 2]})
    assert result.name == "a"


def test_get_column_missing(constructor_eager: ConstructorEager) -> None:
    expected_error: type[Exception]
    backend_name = str(constructor_eager)
    if "polars" in backend_name:
        import polars as pl

        expected_error = pl.exceptions.ColumnNotFoundError
        msg = '"c" not found'
    elif "pyarrow_table" in backend_name:
        expected_error = KeyError
        msg = "'Field \"c\" does not exist in schema'"
    else:  # pandas
        expected_error = KeyError
        msg = "c"

    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(expected_error, match=msg):
        df.get_column("c")


def test_get_column_invalid_type(constructor_eager: ConstructorEager) -> None:
    backend_name = str(constructor_eager)
    if "polars" in backend_name:
        expected_error = TypeError
        msg = "'int' object is not an instance of 'str'"
    elif "pyarrow_table" in backend_name:
        expected_error = TypeError
        msg = "Expected str, got: <class 'int'>"
    else:  # pandas
        expected_error = KeyError
        msg = "0"

    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(expected_error, match=msg):
        df.get_column(0)  # type: ignore[arg-type]


def test_non_string_name() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = pd.DataFrame({0: [1, 2]})
    result = nw.from_native(df, eager_only=True).get_column(0)  # type: ignore[arg-type]
    assert_equal_data({"a": result}, {"a": [1, 2]})
    assert result.name == 0  # type: ignore[comparison-overlap]


def test_get_single_row() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = pd.DataFrame(data)
    result = nw.from_native(df, eager_only=True)[0]
    assert_equal_data(result, {"a": [1], "b": [3]})
