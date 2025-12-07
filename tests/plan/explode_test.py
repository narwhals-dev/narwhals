from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import pytest

import narwhals as nw
import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from narwhals.exceptions import InvalidOperationError, ShapeError
from tests.plan.utils import (
    assert_equal_data,
    assert_equal_series,
    dataframe,
    re_compile,
    series,
)

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
        "l5": [[None, None], [None], [99], [83]],
    }


@pytest.mark.parametrize(
    ("column", "expected_values"),
    [("l2", [None, 3, None, None, 42]), ("l3", [1, 1, 2, 3, None])],
)
def test_explode_frame_single_col(
    column: str, expected_values: list[int | None], data: Data
) -> None:
    result = (
        dataframe(data)
        .with_columns(nwp.col(column).cast(nw.List(nw.Int32())))
        .explode(column)
        .select("a", column)
        .sort("a", column, nulls_last=True)
    )
    expected = {"a": ["w", "x", "x", "y", "z"], column: expected_values}
    assert_equal_data(result, expected)


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
def test_explode_frame_multiple_cols(
    column: str,
    more_columns: Sequence[str],
    expected: dict[str, list[str | int | None]],
    data: Data,
) -> None:
    result = (
        dataframe(data)
        .with_columns(nwp.col(column, *more_columns).cast(nw.List(nw.Int32())))
        .explode(column, *more_columns)
        .select("a", column, *more_columns)
        .sort("a", column, nulls_last=True)
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            ncs.by_index(-1, -2, -3),
            {
                "a": ["w", "x", "x", "y", "z"],
                "l5": [83, None, None, None, 99],
                "l4": [456, 1, 2, 3, 123],
                "l3": [1, 1, 2, 3, None],
            },
        ),
        (
            ncs.matches(r"l[3|5]"),
            {
                "a": ["w", "x", "x", "y", "z"],
                "l3": [1, 1, 2, 3, None],
                "l5": [83, None, None, None, 99],
            },
        ),
    ],
)
def test_explode_frame_selectors(expr: nwp.Selector, expected: Data, data: Data) -> None:
    result = (
        dataframe(data)
        .with_columns(expr.cast(nw.List(nw.Int32())))
        .explode(expr)
        .select("a", expr)
        .sort("a", expr, nulls_last=True)
    )
    assert_equal_data(result, expected)


def test_explode_frame_shape_error(data: Data) -> None:
    with pytest.raises(
        ShapeError, match=r".*exploded columns (must )?have matching element counts"
    ):
        dataframe(data).with_columns(
            nwp.col("l1", "l2", "l3").cast(nw.List(nw.Int32()))
        ).explode(ncs.list())


def test_explode_frame_invalid_operation_error(data: Data) -> None:
    with pytest.raises(
        InvalidOperationError,
        match=re_compile(r"explode.+not supported for.+string.+expected.+list"),
    ):
        dataframe(data).explode("a")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([[1, 2, 3]], [1, 2, 3]),
        ([[1, 2, 3], None], [1, 2, 3, None]),
        ([[1, 2, 3], []], [1, 2, 3, None]),
    ],
)
def test_explode_series_default(values: list[Any], expected: list[Any]) -> None:
    # Based on `test_explode_basic` in https://github.com/pola-rs/polars/issues/25289
    # https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/py-polars/tests/unit/operations/test_explode.py#L465-L505
    result = series(values).explode()
    assert_equal_series(result, expected, "")


@pytest.mark.xfail(
    reason="TODO: 'ArrowExpr' object has no attribute '_evaluated'", raises=AttributeError
)
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([[1, 2, 3], [1, 2], [1, 2]], [1, 2, 3, None, 1, 2]),
        ([[1, 2, 3], [], [1, 2]], [1, 2, 3, None, 1, 2]),
    ],
)
def test_explode_series_default_masked(
    values: list[Any], expected: list[Any]
) -> None:  # pragma: no cover
    result = (
        series(values)
        .to_frame()
        .select(nwp.when(series([True, False, True])).then(nwp.col("")))
        .to_series()
        .explode()
    )
    assert_equal_series(result, expected, "")


DROP_EMPTY: Final = {"empty_as_null": False}
DROP_NULLS: Final = {"keep_nulls": False}
DROP_BOTH: Final = {"empty_as_null": False, "keep_nulls": False}


@pytest.mark.xfail(
    reason="TODO: Implement non-default `Series.explode(...)", raises=NotImplementedError
)
@pytest.mark.parametrize(
    ("values", "kwds", "expected"),
    [
        ([[1, 2, 3]], DROP_BOTH, [1, 2, 3]),
        ([[1, 2, 3], None], DROP_NULLS, [1, 2, 3]),
        ([[1, 2, 3], [None]], DROP_NULLS, [1, 2, 3, None]),
        ([[1, 2, 3], []], DROP_EMPTY, [1, 2, 3]),
        ([[1, 2, 3], [None]], DROP_EMPTY, [1, 2, 3, None]),
    ],
)
def test_explode_series_options(
    values: list[Any], kwds: dict[str, Any], expected: list[Any]
) -> None:  # pragma: no cover
    # Based on `test_explode_basic` in https://github.com/pola-rs/polars/issues/25289
    # https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/py-polars/tests/unit/operations/test_explode.py#L465-L505
    result = series(values).explode(**kwds)
    assert_equal_series(result, expected, "")
