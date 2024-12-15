from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9], "i": [3, 1, 5]}


def test_expr_arg_max_expr(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        # This operation is row-order dependent so we don't support it for Dask
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    df = nw.maybe_set_index(df, "i")
    result = df.select(nw.col("a", "b", "z").arg_max())
    expected = {"a": [1], "b": [2], "z": [2]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("col", "expected"), [("a", 1), ("b", 2), ("z", 2)])
def test_expr_arg_max_series(
    constructor_eager: ConstructorEager, col: str, expected: float
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    series = nw.maybe_set_index(series, index=[1, 0, 9])  # type: ignore[arg-type]
    result = series.arg_max()
    assert_equal_data({col: [result]}, {col: [expected]})
