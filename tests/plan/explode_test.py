from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
from narwhals.exceptions import InvalidOperationError, ShapeError
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    # For context, polars allows to explode multiple columns only if the columns
    # have matching element counts, therefore, l1 and l2 but not l1 and l3 together.
    return {
        "a": ["x", "y", "z", "w"],
        "l1": [[1, 2], None, [None], []],
        "l2": [[3, None], None, [42], []],
        "l3": [[1, 2], [3], [None], [1]],
        "l4": [[1, 2], [3], [123], [456]],
    }


@pytest.mark.xfail(
    reason="TODO:` DataFrame.explode` (single column)", raises=NotImplementedError
)
@pytest.mark.parametrize(
    ("column", "expected_values"),
    [("l2", [None, 3, None, None, 42]), ("l3", [1, 1, 2, 3, None])],
)
def test_explode_single_col(
    column: str, expected_values: list[int | None], data: Data
) -> None:  # pragma: no cover
    result = (
        dataframe(data)
        .with_columns(nwp.col(column).cast(nw.List(nw.Int32())))
        .explode(column)
        .select("a", column)
        .sort("a", column, nulls_last=True)
    )
    expected = {"a": ["w", "x", "x", "y", "z"], column: expected_values}
    assert_equal_data(result, expected)


@pytest.mark.xfail(
    reason="TODO:` DataFrame.explode` (multi column)", raises=NotImplementedError
)
@pytest.mark.parametrize(
    ("column", "more_columns", "expected"),
    [
        (
            "l1",
            ["l2"],
            {
                "a": ["w", "x", "x", "y", "z"],
                "l1": [None, 1, 2, None, None],
                "l2": [None, 3, None, None, 42],
            },
        ),
        (
            "l3",
            ["l4"],
            {
                "a": ["w", "x", "x", "y", "z"],
                "l3": [1, 1, 2, 3, None],
                "l4": [456, 1, 2, 3, 123],
            },
        ),
    ],
)
def test_explode_multiple_cols(
    column: str,
    more_columns: Sequence[str],
    expected: dict[str, list[str | int | None]],
    data: Data,
) -> None:  # pragma: no cover
    result = (
        dataframe(data)
        .with_columns(nwp.col(column, *more_columns).cast(nw.List(nw.Int32())))
        .explode(column, *more_columns)
        .select("a", column, *more_columns)
        .sort("a", column, nulls_last=True)
    )
    assert_equal_data(result, expected)


@pytest.mark.xfail(
    reason="TODO:` DataFrame.explode` (validate shape)",
    raises=(AssertionError, NotImplementedError),
)
def test_explode_shape_error(data: Data) -> None:  # pragma: no cover
    with pytest.raises(
        ShapeError, match=r".*exploded columns (must )?have matching element counts"
    ):
        dataframe(data).with_columns(
            nwp.col("l1", "l2", "l3").cast(nw.List(nw.Int32()))
        ).explode("l1", "l3")


@pytest.mark.xfail(
    reason="TODO:` DataFrame.explode` (validate dtype)",
    raises=(AssertionError, NotImplementedError),
)
def test_explode_invalid_operation_error(data: Data) -> None:  # pragma: no cover
    with pytest.raises(
        InvalidOperationError, match="`explode` operation not supported for dtype"
    ):
        dataframe(data).explode("a")
