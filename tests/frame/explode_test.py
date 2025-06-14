from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError, ShapeError
from tests.utils import PANDAS_VERSION, POLARS_VERSION, Constructor, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Sequence

# For context, polars allows to explode multiple columns only if the columns
# have matching element counts, therefore, l1 and l2 but not l1 and l3 together.
data = {
    "a": ["x", "y", "z", "w"],
    "l1": [[1, 2], None, [None], []],
    "l2": [[3, None], None, [42], []],
    "l3": [[1, 2], [3], [None], [1]],
    "l4": [[1, 2], [3], [123], [456]],
}


@pytest.mark.parametrize(
    ("column", "expected_values"),
    [
        ("l2", [None, 3, None, None, 42]),
        ("l3", [1, 1, 2, 3, None]),  # fast path for arrow
    ],
)
def test_explode_single_col(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    column: str,
    expected_values: list[int | None],
) -> None:
    if any(
        backend in str(constructor)
        for backend in ("dask", "modin", "cudf", "pyarrow_table")
    ):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor) and PANDAS_VERSION < (2, 2):
        pytest.skip()

    result = (
        nw.from_native(constructor(data))
        .with_columns(nw.col(column).cast(nw.List(nw.Int32())))
        .explode(column)
        .select("a", column)
        .sort("a")
    )
    expected = {"a": ["w", "x", "x", "y", "z"], column: expected_values}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("columns", "more_columns", "expected"),
    [
        (
            "l1",
            ["l2"],
            {
                "a": ["x", "x", "y", "z", "w"],
                "l1": [1, 2, None, None, None],
                "l2": [3, None, None, 42, None],
            },
        ),
        (
            "l3",
            ["l4"],
            {
                "a": ["x", "x", "y", "z", "w"],
                "l3": [1, 2, 3, None, 1],
                "l4": [1, 2, 3, 123, 456],
            },
        ),
    ],
)
def test_explode_multiple_cols(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    columns: str | Sequence[str],
    more_columns: Sequence[str],
    expected: dict[str, list[str | int | None]],
) -> None:
    if any(
        backend in str(constructor)
        for backend in (
            "dask",
            "modin",
            "cudf",
            "pyarrow_table",
            "duckdb",
            "pyspark",
            "ibis",
        )
    ):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor) and PANDAS_VERSION < (2, 2):
        pytest.skip()

    result = (
        nw.from_native(constructor(data))
        .with_columns(nw.col(columns, *more_columns).cast(nw.List(nw.Int32())))
        .explode(columns, *more_columns)
        .select("a", columns, *more_columns)
    )
    assert_equal_data(result, expected)


def test_explode_shape_error(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(
        backend in str(constructor)
        for backend in ("dask", "modin", "cudf", "pyarrow_table")
    ):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor) and PANDAS_VERSION < (2, 2):
        pytest.skip()

    with pytest.raises(
        (ShapeError, NotImplementedError),
        match=r".*exploded columns (must )?have matching element counts",
    ):
        _ = (
            nw.from_native(constructor(data))
            .lazy()
            .with_columns(nw.col("l1", "l2", "l3").cast(nw.List(nw.Int32())))
            .explode("l1", "l3")
            .collect()
        )


def test_explode_invalid_operation_error(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "dask")):
        request.applymarker(pytest.mark.xfail)

    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 6):
        pytest.skip()

    with pytest.raises(
        InvalidOperationError, match="`explode` operation not supported for dtype"
    ):
        _ = nw.from_native(constructor(data)).lazy().explode("a").collect()
