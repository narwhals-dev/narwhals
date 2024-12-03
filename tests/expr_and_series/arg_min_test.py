from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

# Sample data
data = {"a": [1, 3, 2], "b": [4, 4, 7], "z": [7.0, 8, 9]}


@pytest.mark.parametrize(
    "expr",
    [
        # Assuming that nw.col returns a column expression that supports arg_min as a method
        nw.col("a", "b", "z").arg_max(),
    ],
)
def test_expr_argmin_expr(constructor: Constructor, expr: nw.Expr) -> None:
    # Convert native data to a DataFrame using the constructor
    df = nw.from_native(constructor(data))
    # Select columns using the expression
    result = df.select(expr)
    # Define the expected result, assuming the minimum index is 0 for all columns
    expected = {"a": [0], "b": [0], "z": [0]}
    # Assert that the result data matches the expected data
    assert_equal_data(result, expected)
