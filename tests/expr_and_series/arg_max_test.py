from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw  # Assuming this is a placeholder for your abstraction layer
from tests.utils import PANDAS_VERSION
from tests.utils import PYARROW_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

# Define sample data for testing
data = {"a": [1, 3, None, 2]}

# Expected results for arg_max
expected = {
    "arg_max": [1, 1, None, 1],
}


def test_arg_max_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    # Handle version-specific expected failures
    if PYARROW_VERSION < (13, 0, 0) and "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    if (PANDAS_VERSION < (2, 1) or PYARROW_VERSION < (13,)) and "pandas_pyarrow" in str(
        constructor
    ):
        request.applymarker(pytest.mark.xfail)

    # Create a DataFrame from the constructor
    df = nw.from_native(constructor(data))

    # Test the arg_max expression
    result = df.select(
        nw.col("a").arg_max().alias("arg_max"),
    )

    # Assert that the result matches the expected data
    assert_equal_data(result, {"arg_max": expected["arg_max"]})


def test_arg_max_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    # Version-specific xfail setup
    if PYARROW_VERSION < (13, 0, 0) and "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    if (PANDAS_VERSION < (2, 1) or PYARROW_VERSION < (13,)) and "pandas_pyarrow" in str(
        constructor_eager
    ):
        request.applymarker(pytest.mark.xfail)

    # Create a DataFrame for eager computation
    df = nw.from_native(constructor_eager(data), eager_only=True)

    # Test arg_max on series level
    result = df.select(
        arg_max=df["a"].arg_max(),
    )

    # Assert that the data matches the expected output
    assert_equal_data(result, {"arg_max": expected["arg_max"]})
