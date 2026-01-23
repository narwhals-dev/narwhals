from __future__ import annotations

# ruff: noqa: F401
import copy
from typing import TYPE_CHECKING, Any, Final

import pytest

import narwhals as nw
import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from narwhals.exceptions import ColumnNotFoundError, InvalidOperationError, ShapeError
from tests.plan.utils import (
    assert_equal_data,
    assert_equal_series,
    dataframe,
    re_compile,
    series,
)

if TYPE_CHECKING:
    from narwhals._plan.typing import ColumnNameOrSelector, OneOrIterable
    from tests.conftest import Data

pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.compute as pc


@pytest.fixture
def data_1() -> Data:
    return {
        "before": ["foo", "bar"],
        "t_a": [1, 2],
        "t_b": ["a", "b"],
        "t_c": [True, None],
        "t_d": [[1, 2], [3]],
        "after": ["baz", "womp"],
    }


XFAIL_NOT_IMPL_MULTI_STRUCT = pytest.mark.xfail(
    reason="TODO: ArrowDataFrame.unnest(columns=[..., ...])"
)


def pyarrow_struct(native: pa.Table, columns: list[str]) -> pa.StructArray:
    return pc.make_struct(*native.select(columns).columns, field_names=columns)


@pytest.mark.parametrize(
    "columns",
    [["t_a"], ["t_a", "t_b"], ["t_a", "t_b", "t_c", "t_d"]],
    ids=["1-column", "2-column", "4-column"],
)
def test_unnest_frame_single_struct(data_1: Data, columns: list[str]) -> None:
    expected = copy.deepcopy(data_1)
    table = pa.Table.from_pydict(data_1)
    table_w_struct = table.drop(columns).add_column(
        1, "t_struct", pyarrow_struct(table, columns)
    )

    df = nwp.DataFrame.from_native(table_w_struct)
    assert_equal_data(df.unnest("t_struct"), expected)
    assert_equal_data(df.unnest(ncs.struct()), expected)
    assert_equal_data(df.unnest(nwp.nth(1).meta.as_selector()), expected)


@XFAIL_NOT_IMPL_MULTI_STRUCT
def test_unnest_frame_multi_struct(data_1: Data) -> None:  # pragma: no cover
    expected = copy.deepcopy(data_1)
    table = pa.Table.from_pydict(data_1)
    columns_1 = ["t_a", "t_b"]
    columns_2 = ["t_c", "t_d"]
    name_1 = "t_struct_1"
    name_2 = "t_struct_2"
    table_w_structs = (
        table.drop([*columns_1, *columns_2])
        .add_column(1, name_1, pyarrow_struct(table, columns_1))
        .add_column(2, name_2, pyarrow_struct(table, columns_2))
    )

    df = nwp.DataFrame.from_native(table_w_structs)
    assert_equal_data(df.unnest(name_1, name_2), expected)
    assert_equal_data(df.unnest(ncs.struct()), expected)
    assert_equal_data(df.unnest(ncs.by_index(1, 2)), expected)


def test_unnest_frame_invalid_operation_error(data_1: Data) -> None:
    df = dataframe(data_1)
    with pytest.raises(
        InvalidOperationError,
        match=re_compile(r"unnest.+not supported for.+string.+expected.+struct"),
    ):
        df.unnest("before")

    with pytest.raises(InvalidOperationError):
        df.unnest(ncs.last().first())


@pytest.mark.parametrize(
    "columns", [ncs.array(), "hello", nwp.nth(29).meta.as_selector(), ["a", "b", "A"]]
)
def test_unnest_frame_column_not_found_error(
    data_1: Data, columns: OneOrIterable[ColumnNameOrSelector]
) -> None:
    df = dataframe(data_1)
    with pytest.raises(ColumnNotFoundError):
        df.unnest(columns)
