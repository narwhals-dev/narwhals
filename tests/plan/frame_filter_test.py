from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("pyarrow")

import narwhals._plan as nwp
from narwhals.exceptions import ColumnNotFoundError, InvalidOperationError, ShapeError
from tests.plan.utils import assert_equal_data, dataframe, series

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture
def data() -> Data:
    return {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}


@pytest.mark.parametrize(
    "predicate",
    [[False, True, True], series([False, True, True]), nwp.col("a") > 1],
    ids=["list[bool]", "Series", "Expr"],
)
def test_filter_single(
    data: Data, predicate: list[bool] | nwp.Series[Any] | nwp.Expr
) -> None:
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    result = dataframe(data).filter(predicate)
    assert_equal_data(result, expected)


@pytest.mark.xfail(reason=("Not sure why this isn't allowed on `main`"))
def test_filter_raise_on_agg_predicate(data: Data) -> None:
    df = dataframe(data)
    with pytest.raises(InvalidOperationError):
        df.filter(nwp.col("a").max() > 2)


def test_filter_raise_on_shape_mismatch(data: Data) -> None:
    df = dataframe(data)
    with pytest.raises(ShapeError):
        df.filter(nwp.col("b").filter(nwp.col("b") < 6))


def test_filter_with_constraints() -> None:
    df = dataframe({"a": [1, 3, 2], "b": [4, 4, 6]})
    result_scalar = df.filter(a=3)
    expected_scalar = {"a": [3], "b": [4]}
    assert_equal_data(result_scalar, expected_scalar)
    result_expr = df.filter(a=nwp.col("b") // 3)
    expected_expr = {"a": [1, 2], "b": [4, 6]}
    assert_equal_data(result_expr, expected_expr)


def test_filter_missing_column() -> None:
    df = dataframe({"a": [1, 2], "b": [3, 4]})
    msg = (
        r"The following columns were not found: \[.*\]"
        r"\n\nHint: Did you mean one of these columns: \['a', 'b'\]?"
    )
    with pytest.raises(ColumnNotFoundError, match=msg):
        df.filter(c=5)


@pytest.mark.xfail(
    reason="Need a resolution to https://github.com/narwhals-dev/narwhals/issues/3182"
)
def test_filter_mask_mixed() -> None:
    df = dataframe({"a": range(5), "b": [2, 2, 4, 2, 4]})
    mask = [True, False, True, True, False]
    mask_2 = [True, True, False, True, False]
    expected_mask_only = {"a": [0, 2, 3], "b": [2, 4, 2]}
    expected_mixed = {"a": [0, 3], "b": [2, 2]}

    result = df.filter(mask)
    assert_equal_data(result, expected_mask_only)

    with pytest.raises(
        ColumnNotFoundError, match=re.escape("not found: ['c', 'd', 'e', 'f', 'g']")
    ):
        df.filter(mask, c=1, d=2, e=3, f=4, g=5)  # type: ignore[arg-type]

    # NOTE: Everything from here is currently undefined
    result = df.filter(mask, b=2)  # type: ignore[arg-type]
    assert_equal_data(result, expected_mixed)

    result = df.filter(mask, nwp.col("b") == 2)  # type: ignore[arg-type]
    assert_equal_data(result, expected_mixed)

    result = df.filter(mask, mask_2)  # type: ignore[arg-type]
    assert_equal_data(result, expected_mixed)

    result = df.filter(mask, series(mask_2))  # type: ignore[arg-type]
    assert_equal_data(result, expected_mixed)

    result = df.filter(mask, nwp.col("b") != 4, b=2)  # type: ignore[arg-type]
    assert_equal_data(result, expected_mixed)
