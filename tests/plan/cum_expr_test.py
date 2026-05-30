from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import pytest

import narwhals._plan as nwp
from narwhals.exceptions import InvalidOperationError
from tests.plan.utils import DataFrame, assert_equal_data, xfail_polars_over_order_by

if TYPE_CHECKING:
    from pytest import FixtureRequest

    from tests.conftest import Data
    from tests.plan.utils import DataFrame

Fn: TypeAlias = Callable[[nwp.Expr], nwp.Expr]

cum_count: Final = nwp.Expr.cum_count
cum_max: Final = nwp.Expr.cum_max
cum_min: Final = nwp.Expr.cum_min
cum_prod: Final = nwp.Expr.cum_prod
cum_sum: Final = nwp.Expr.cum_sum

cum_count_rev: Final = partial(nwp.Expr.cum_count, reverse=True)
cum_max_rev: Final = partial(nwp.Expr.cum_max, reverse=True)
cum_min_rev: Final = partial(nwp.Expr.cum_min, reverse=True)
cum_prod_rev: Final = partial(nwp.Expr.cum_prod, reverse=True)
cum_sum_rev: Final = partial(nwp.Expr.cum_sum, reverse=True)

a = nwp.col("a")


def id_func(obj: Callable[..., Any] | Any) -> str:
    if callable(obj):
        if isinstance(obj, partial):
            return f"{obj.func.__name__}-reverse"
        name: str = obj.__name__
        return name
    return ""


# TODO @dangotbanned: Figure out what's going on
def xfail_pyarrow_bug(dataframe: DataFrame, request: FixtureRequest) -> None:
    dataframe.xfail(
        request,
        dataframe.is_pyarrow(),
        reason="BUG: pyarrow has incorrect results for `cum_*.over(order_by=...)`",
        raises=AssertionError,
    )


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (cum_count, [1, 2, 2, 3]),
        (cum_count_rev, [3, 2, 1, 1]),
        (cum_max, [1, 3, None, 3]),
        (cum_max_rev, [3, 3, None, 2]),
        (cum_min, [1, 1, None, 1]),
        (cum_min_rev, [1, 2, None, 2]),
        (cum_prod, [1, 3, None, 6]),
        (cum_prod_rev, [6, 6, None, 2]),
        (cum_sum, [1, 4, None, 6]),
        (cum_sum_rev, [6, 5, None, 2]),
    ],
    ids=id_func,
)
def test_cumulative_eager(
    fn: Fn, expected: list[int | None], dataframe: DataFrame
) -> None:
    data = {"a": [1, 3, None, 2]}
    expr = fn(a)
    result = dataframe(data).select(expr)
    assert_equal_data(result, {"a": expected})


@pytest.fixture
def data_over() -> Data:
    return {
        "a": [None, 2, 3, 1, 2, 3, 4],
        "b": [1, -1, 3, 2, 5, 0, None],
        "i": [0, 1, 2, 3, 4, 5, 6],
        # NOTE: The original dataset doesn't test partitions
        # everything is `1`
        "g": [2, 2, 0, 0, 2, None, 1],
    }


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (cum_max, [None, 4, 4, 4, 4, 4, 4]),
        (cum_max_rev, [None, 3, 3, 3, 2, 3, 4]),
        (cum_min, [None, 2, 1, 1, 1, 2, 4]),
        (cum_min_rev, [None, 1, 2, 1, 2, 1, 1]),
        (cum_sum, [None, 6, 13, 10, 15, 9, 4]),
        (cum_sum_rev, [None, 11, 5, 6, 2, 9, 15]),
    ],
    ids=id_func,
)
def test_cumulative_order_by(
    data_over: Data,
    fn: Fn,
    expected: list[int | None],
    dataframe: DataFrame,
    request: FixtureRequest,
) -> None:
    xfail_polars_over_order_by(dataframe, request)
    expr = fn(a).over(order_by="b")
    xfail_pyarrow_bug(dataframe, request)
    result = dataframe(data_over).with_columns(expr).sort("i")
    assert_equal_data(result, data_over | {"a": expected})


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (cum_count, [1, 1, 2, 1, 2, 1, 1]),
        (cum_count_rev, [1, 2, 1, 2, 1, 1, 1]),
        (cum_max, [None, 2, 3, 1, 2, 3, 4]),
        (cum_max_rev, [None, 2, 3, 3, 2, 3, 4]),
        (cum_min, [None, 2, 1, 1, 2, 3, 4]),
        (cum_min_rev, [None, 2, 3, 1, 2, 3, 4]),
        (cum_prod, [None, 2, 3, 1, 4, 3, 4]),
        (cum_prod_rev, [None, 4, 3, 3, 2, 3, 4]),
        (cum_sum, [None, 2, 4, 1, 4, 3, 4]),
        (cum_sum_rev, [None, 4, 3, 4, 2, 3, 4]),
    ],
    ids=id_func,
)
def test_cumulative_partition_by_order_by(
    data_over: Data,
    fn: Fn,
    expected: list[int | None],
    dataframe: DataFrame,
    request: FixtureRequest,
) -> None:
    xfail_polars_over_order_by(dataframe, request)
    dataframe.xfail(
        request,
        dataframe.is_pyarrow(),
        reason="`cum_*.over(*partition_by, order_by=...)` is not implemented for `pyarrow`",
        raises=(NotImplementedError, InvalidOperationError),
    )
    expr = fn(a).over("g", order_by="b")
    result = dataframe(data_over).with_columns(expr).sort("i")
    assert_equal_data(result, data_over | {"a": expected})


def test_shift_cum_sum(dataframe: DataFrame) -> None:
    data = {"a": [1, 2, 3, 4, 5], "i": [0, 5, 2, 3, 2]}
    result = dataframe(data).select(a.shift(1).cum_sum())
    assert_equal_data(result, {"a": [None, 1, 3, 6, 10]})


def test_shift_cum_sum_order_by(dataframe: DataFrame, request: FixtureRequest) -> None:
    xfail_polars_over_order_by(dataframe, request)
    xfail_pyarrow_bug(dataframe, request)
    data = {"a": [1, 2, 3, 4, 5], "i": [0, 5, 2, 3, 2]}
    result = dataframe(data).select(a.shift(1).cum_sum().over(order_by="i"))
    assert_equal_data(result, {"a": [None, 13, 1, 9, 4]})
