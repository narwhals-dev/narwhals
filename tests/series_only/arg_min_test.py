from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}


@pytest.mark.parametrize(("col", "expected"), [("a", 0), ("b", 0), ("z", 0)])
def test_arg_min_series(
    nw_eager_constructor: ConstructorEager,
    col: str,
    expected: float,
    request: pytest.FixtureRequest,
) -> None:
    if "modin" in str(nw_eager_constructor):
        # TODO(unassigned): bug in modin?
        return
    if "cudf" in str(nw_eager_constructor):
        # not implemented yet
        request.applymarker(pytest.mark.xfail)
    series = nw.from_native(nw_eager_constructor(data), eager_only=True)[col]
    series = nw.maybe_set_index(series, index=[1, 0, 9])  # type: ignore[arg-type]
    result = series.arg_min()
    assert_equal_data({col: [result]}, {col: [expected]})
