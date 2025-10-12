from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


def test_floor(constructor: Constructor) -> None:
    data = {"a": [1.12345, 2.56789, 3.901234, -0.5]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    expected_data = {"a": [1.0, 2.0, 3.0, -1.0]}
    result_frame = df.select(nw.col("a").floor())

    if "pandas_pyarrow" in str(constructor):
        pandas_input = df_raw["a"]  # type:ignore[index]
        assert pandas_input.dtype == result_frame["a"].to_native().dtype

    assert_equal_data(result_frame, expected_data)


def test_ceil(constructor: Constructor) -> None:
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
    x = df["a"]
    result_series = x.floor()

    assert_equal_data({"a": result_series}, expected_data)


def test_ceil_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1.12345, 2.56789, 3.901234, -0.5]}
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw, eager_only=True)

    expected_data = {"a": [2.0, 3.0, 4.0, 0.0]}
    result_series = df["a"].ceil()

    assert_equal_data({"a": result_series}, expected_data)
