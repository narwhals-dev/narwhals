from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("pyarrow")

import narwhals._plan as nwp
from narwhals.exceptions import ColumnNotFoundError, InvalidOperationError
from tests.plan.utils import assert_equal_data, dataframe, series

if TYPE_CHECKING:
    from tests.conftest import Data


XFAIL_DATAFRAME_FILTER = pytest.mark.xfail(
    reason=("Not implemented `DataFrame.filter` yet"), raises=NotImplementedError
)


@pytest.fixture
def data() -> Data:
    return {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}


@XFAIL_DATAFRAME_FILTER
def test_filter(data: Data) -> None:  # pragma: no cover
    result = dataframe(data).filter(nwp.col("a") > 1)
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    assert_equal_data(result, expected)


# NOTE: On `main`, this test uses `Series.__gt__`
def test_filter_with_series(data: Data) -> None:
    predicate = series([False, True, True])
    result = dataframe(data).filter(predicate)
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    assert_equal_data(result, expected)


def test_filter_with_boolean_list(data: Data) -> None:
    result = dataframe(data).filter([False, True, True])
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    assert_equal_data(result, expected)


@XFAIL_DATAFRAME_FILTER
def test_filter_raise_on_agg_predicate(data: Data) -> None:  # pragma: no cover
    df = dataframe(data)
    with pytest.raises(InvalidOperationError):
        df.filter(nwp.col("a").max() > 2)


@XFAIL_DATAFRAME_FILTER
def test_filter_raise_on_shape_mismatch(data: Data) -> None:  # pragma: no cover
    df = dataframe(data)
    with pytest.raises(InvalidOperationError):
        df.filter(nwp.col("b").unique() > 2)


@XFAIL_DATAFRAME_FILTER
def test_filter_with_constraints() -> None:  # pragma: no cover
    df = dataframe({"a": [1, 3, 2], "b": [4, 4, 6]})
    result_scalar = df.filter(a=3)
    expected_scalar = {"a": [3], "b": [4]}
    assert_equal_data(result_scalar, expected_scalar)
    result_expr = df.filter(a=nwp.col("b") // 3)
    expected_expr = {"a": [1, 2], "b": [4, 6]}
    assert_equal_data(result_expr, expected_expr)


@XFAIL_DATAFRAME_FILTER
def test_filter_missing_column() -> None:  # pragma: no cover
    df = dataframe({"a": [1, 2], "b": [3, 4]})
    msg = (
        r"The following columns were not found: \[.*\]"
        r"\n\nHint: Did you mean one of these columns: \['a', 'b'\]?"
    )
    with pytest.raises(ColumnNotFoundError, match=msg):
        df.filter(c=5)
