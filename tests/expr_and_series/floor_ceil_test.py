from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data


def test_floor_expr(constructor: Constructor) -> None:
    data = {"a": [1.12345, 2.56789, 3.901234, -0.5]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    expected_data = {"a": [1.0, 2.0, 3.0, -1.0]}
    result_frame = df.select(nw.col("a").floor())

    assert_equal_data(result_frame, expected_data)


def test_ceil_expr(constructor: Constructor) -> None:
    data = {"a": [1.12345, 2.56789, 3.901234, -0.5]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    expected_data = {"a": [2.0, 3.0, 4.0, 0.0]}
    result_frame = df.select(nw.col("a").ceil())
    assert_equal_data(result_frame, expected_data)


def test_floor_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1.12345, 2.56789, 3.901234, -0.5]}
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw, eager_only=True)

    expected_data = {"a": [1.0, 2.0, 3.0, -1.0]}
    result_series = df["a"].floor()

    assert_equal_data({"a": result_series}, expected_data)


def test_ceil_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1.12345, 2.56789, 3.901234, -0.5]}
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw, eager_only=True)

    expected_data = {"a": [2.0, 3.0, 4.0, 0.0]}
    result_series = df["a"].ceil()

    assert_equal_data({"a": result_series}, expected_data)


def test_floor_dtype_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s = pd.Series([1.0, 2.2], name="a", dtype="float32", index=[8, 7])
    result = nw.from_native(s, series_only=True).floor().to_native()
    expected = pd.Series([1.0, 2.0], name="a", dtype="float32", index=[8, 7])
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="nullable types require pandas2+")
def test_floor_dtype_pandas_nullable() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s = pd.Series([1.0, None, 2.2], name="a", dtype="Float32", index=[8, 7, 6])
    result = nw.from_native(s, series_only=True).floor().to_native()
    expected = pd.Series([1.0, None, 2.0], name="a", dtype="Float32", index=[8, 7, 6])
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(PANDAS_VERSION < (2, 1, 0), reason="nullable types require pandas2+")
def test_floor_dtype_pandas_pyarrow() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    s = pd.Series([1.0, None, 2.2], name="a", dtype="Float32[pyarrow]", index=[8, 7, 6])
    result = nw.from_native(s, series_only=True).floor().to_native()
    expected = pd.Series(
        [1.0, None, 2.0], name="a", dtype="Float32[pyarrow]", index=[8, 7, 6]
    )
    pd.testing.assert_series_equal(result, expected)


def test_ceil_dtype_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s = pd.Series([1.0, 2.2], name="a", dtype="float32", index=[8, 7])
    result = nw.from_native(s, series_only=True).ceil().to_native()
    expected = pd.Series([1.0, 3.0], name="a", dtype="float32", index=[8, 7])
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="nullable types require pandas2+")
def test_ceil_dtype_pandas_nullable() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s = pd.Series([1.0, None, 2.2], name="a", dtype="Float32", index=[8, 7, 6])
    result = nw.from_native(s, series_only=True).ceil().to_native()
    expected = pd.Series([1.0, None, 3.0], name="a", dtype="Float32", index=[8, 7, 6])
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(PANDAS_VERSION < (2, 1, 0), reason="nullable types require pandas2+")
def test_ceil_dtype_pandas_pyarrow() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    s = pd.Series([1.0, None, 2.2], name="a", dtype="Float32[pyarrow]", index=[8, 7, 6])
    result = nw.from_native(s, series_only=True).ceil().to_native()
    expected = pd.Series(
        [1.0, None, 3.0], name="a", dtype="Float32[pyarrow]", index=[8, 7, 6]
    )
    pd.testing.assert_series_equal(result, expected)
