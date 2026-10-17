from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0], "i": [3, 1, 5]}


@pytest.mark.parametrize(("col", "expected"), [("a", 1), ("b", 2), ("z", 2)])
def test_arg_max_series(
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
    result = series.arg_max()
    assert_equal_data({col: [result]}, {col: [expected]})


@pytest.mark.parametrize("empty", [False, True], ids=["all_null", "empty"])
def test_arg_max_series_no_valid_values(
    constructor_eager: ConstructorEager, *, empty: bool
) -> None:
    df = nw.from_native(constructor_eager({"a": [1.0, 2.0]}), eager_only=True)
    if empty:
        series = df.filter(nw.col("a") < 0)["a"]
    else:
        series = df.with_columns(a=nw.when(nw.col("a") < 0).then(nw.col("a")))["a"]
    assert series.arg_max() is None


@pytest.mark.parametrize(
    ("values", "expected"),
    [([None, 3.0, 3.0, None], 1), ([None, -3.0, 2.0, None], 2)],
    ids=["ties_between_nulls", "signed_between_nulls"],
)
def test_arg_max_series_skips_nulls(
    constructor_eager: ConstructorEager, values: list[float | None], expected: int
) -> None:
    series = nw.from_native(constructor_eager({"a": values}), eager_only=True)["a"]
    assert series.arg_max() == expected
