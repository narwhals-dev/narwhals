from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [0, 0, 2, -1], "b": [1, 3, 2, None]}


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, {"b": [3, 2, 1, float("nan")]}),
        (True, False, {"b": [float("nan"), 3, 2, 1]}),
        (False, True, {"b": [1, 2, 3, float("nan")]}),
        (False, False, {"b": [float("nan"), 1, 2, 3]}),
    ],
)
def test_sort_single_expr(
    constructor: Constructor,
    descending: bool,  # noqa: FBT001
    nulls_last: bool,  # noqa: FBT001
    expected: dict[str, float],
    request: pytest.FixtureRequest,
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("b").sort(descending=descending, nulls_last=nulls_last))
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, {"a": [0, 0, 2, -1], "b": [3, 2, 1, float("nan")]}),
        (True, False, {"a": [0, 0, 2, -1], "b": [float("nan"), 3, 2, 1]}),
        (False, True, {"a": [0, 0, 2, -1], "b": [1, 2, 3, float("nan")]}),
        (False, False, {"a": [0, 0, 2, -1], "b": [float("nan"), 1, 2, 3]}),
    ],
)
def test_sort_multiple_expr(
    constructor: Constructor,
    descending: bool,  # noqa: FBT001
    nulls_last: bool,  # noqa: FBT001
    expected: dict[str, float],
    request: pytest.FixtureRequest,
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(
        "a",
        nw.col("b").sort(descending=descending, nulls_last=nulls_last),
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, [3, 2, 1, float("nan")]),
        (True, False, [float("nan"), 3, 2, 1]),
        (False, True, [1, 2, 3, float("nan")]),
        (False, False, [float("nan"), 1, 2, 3]),
    ],
)
def test_sort_series(
    constructor_eager: ConstructorEager,
    descending: bool,  # noqa: FBT001
    nulls_last: bool,  # noqa: FBT001
    expected: dict[str, float],
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["b"]
    result = series.sort(descending=descending, nulls_last=nulls_last)
    assert_equal_data({"b": result}, {"b": expected})
