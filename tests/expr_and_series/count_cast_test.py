from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_count_cast(constructor: Constructor) -> None:
    # Create test data with nulls to verify count behavior
    data = {"a": [1, 3, 2], "b": [4, None, 6], "z": [7.0, None, None]}
    df = nw.from_native(constructor(data))

    # Test count with cast using DType class (preferred way)
    result = df.select(nw.col("a", "b", "z").count().cast(nw.Int64))
    expected = {"a": [3], "b": [2], "z": [1]}
    assert_equal_data(result, expected)

    # Test count with cast using string dtype (for backward compatibility)
    result = df.select(nw.col("a", "b", "z").count().cast("int64"))  # type: ignore[arg-type]
    assert_equal_data(result, expected)
