from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}


def test_expr_arg_min_expr(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor_eager):
        # not implemented yet
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a", "b", "z").arg_min())
    expected = {"a": [0], "b": [0], "z": [0]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("col", "expected"), [("a", 0), ("b", 0), ("z", 0)])
def test_expr_arg_min_series(
    constructor_eager: ConstructorEager,
    col: str,
    expected: float,
    request: pytest.FixtureRequest,
) -> None:
    if "modin" in str(constructor_eager):
        # TODO(unassigned): bug in modin?
        return
    if "cudf" in str(constructor_eager):
        # not implemented yet
        request.applymarker(pytest.mark.xfail)
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    series = nw.maybe_set_index(series, index=[1, 0, 9])  # type: ignore[arg-type]
    result = series.arg_min()
    assert_equal_data({col: [result]}, {col: [expected]})
