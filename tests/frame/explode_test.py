from __future__ import annotations

from typing import Sequence

import pytest
from polars.exceptions import ShapeError as PlShapeError

import narwhals.stable.v1 as nw
from narwhals.exceptions import ShapeError
from tests.utils import Constructor
from tests.utils import assert_equal_data

# For context, polars allows to explode multiple columns only if the columns
# have matching element counts, therefore, l1 and l2 but not l1 and l3 together.
data = {
    "a": ["x", "y", "z", "w"],
    "l1": [[1, 2], None, [None], []],
    "l2": [[3, None], None, [42], []],
    "l3": [[1, 2], [3], [None], [1]],
}


@pytest.mark.parametrize(
    ("columns", "expected_values"),
    [
        ("l2", [3, None, None, 42, None]),
        ("l3", [1, 2, 3, None, 1]),  # fast path for arrow
    ],
)
def test_explode_single_col(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    columns: str,
    expected_values: list[int | None],
) -> None:
    if any(backend in str(constructor) for backend in ("dask", "modin", "cudf")):
        request.applymarker(pytest.mark.xfail)

    result = (
        nw.from_native(constructor(data))
        .with_columns(nw.col("l1", "l2", "l3").cast(nw.List(nw.Int32())))
        .explode(columns)
        .select("a", columns)
    )
    expected = {"a": ["x", "x", "y", "z", "w"], columns: expected_values}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("columns", "more_columns"),
    [
        ("l1", ["l2"]),
        (["l1", "l2"], []),
    ],
)
def test_explode_multiple_cols(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    columns: str | Sequence[str],
    more_columns: Sequence[str],
) -> None:
    if any(backend in str(constructor) for backend in ("dask", "modin", "cudf")):
        request.applymarker(pytest.mark.xfail)

    result = (
        nw.from_native(constructor(data))
        .with_columns(nw.col("l1", "l2", "l3").cast(nw.List(nw.Int32())))
        .explode(columns, *more_columns)
        .select("a", "l1", "l2")
    )
    expected = {
        "a": ["x", "x", "y", "z", "w"],
        "l1": [1, 2, None, None, None],
        "l2": [3, None, None, 42, None],
    }
    assert_equal_data(result, expected)


def test_explode_exception(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(backend in str(constructor) for backend in ("dask", "modin", "cudf")):
        request.applymarker(pytest.mark.xfail)

    with pytest.raises(
        (ShapeError, PlShapeError),
        match="exploded columns must have matching element counts",
    ):
        _ = (
            nw.from_native(constructor(data))
            .lazy()
            .with_columns(nw.col("l1", "l2", "l3").cast(nw.List(nw.Int32())))
            .explode("l1", "l3")
            .collect()
        )
