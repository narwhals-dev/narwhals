from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pytest

pytest.importorskip("pyarrow")

from collections.abc import Mapping

import narwhals as nw
import narwhals._plan as nwp
from narwhals._plan import selectors as ncs
from narwhals._utils import zip_strict
from narwhals.exceptions import InvalidOperationError
from narwhals.typing import NonNestedLiteral
from tests.plan.utils import assert_equal_data, dataframe, first, last, re_compile

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from _pytest.mark import ParameterSet
    from typing_extensions import TypeAlias

    from narwhals._plan.typing import IntoExprColumn, OneOrIterable
    from tests.conftest import Data


@pytest.fixture
def data() -> Data:
    return {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "i": [0, 1, 2, 3, 4],
    }


@pytest.fixture
def data_with_null(data: Data) -> Data:
    return data | {"b": [1, 2, None, 5, 3]}


@pytest.fixture
def data_alt() -> Data:
    return {"a": [3, 5, 1, 2, None], "b": [0, 1, 3, 2, 1], "c": [9, 1, 2, 1, 1]}


@pytest.mark.parametrize(
    "partition_by",
    [
        "a",
        ["a"],
        nwp.nth(0),
        ncs.first(),
        ncs.string(),
        ncs.by_dtype(nw.String),
        ncs.by_name("a"),
        ncs.matches(r"a"),
        ncs.all() - ncs.numeric(),
    ],
)
def test_over_single(data: Data, partition_by: OneOrIterable[IntoExprColumn]) -> None:
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "i": [0, 1, 2, 3, 4],
        "c_max": [5, 5, 3, 3, 3],
    }
    result = (
        dataframe(data)
        .with_columns(c_max=nwp.col("c").max().over(partition_by))
        .sort("i")
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    "partition_by",
    [
        ("a", "b"),
        [nwp.col("a"), nwp.col("b")],
        [nwp.nth(0), nwp.nth(1)],
        nwp.col("a", "b"),
        nwp.nth(0, 1),
        ncs.by_name("a", "b"),
        ncs.matches(r"a|b"),
        ncs.all() - ncs.by_name(["c", "i"]),
    ],
    ids=[
        "tuple[str]",
        "col-col",
        "nth-nth",
        "cols",
        "index_columns",
        "by_name",
        "matches",
        "binary_selector",
    ],
)
def test_over_multiple(data: Data, partition_by: OneOrIterable[IntoExprColumn]) -> None:
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "i": [0, 1, 2, 3, 4],
        "c_min": [5, 4, 1, 2, 1],
    }
    result = (
        dataframe(data)
        .with_columns(c_min=nwp.col("c").min().over(partition_by))
        .sort("i")
    )
    assert_equal_data(result, expected)


# NOTE: Not planned
@pytest.mark.xfail(
    reason="Native `pyarrow` `group_by` isn't enough", raises=InvalidOperationError
)
def test_over_cum_sum_partition_by(data_with_null: Data) -> None:  # pragma: no cover
    df = dataframe(data_with_null)
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cum_sum": [1, 3, None, 5, 8],
        "c_cum_sum": [5, 9, 3, 5, 6],
    }

    result = (
        df.with_columns(nwp.col("b", "c").cum_sum().over("a").name.suffix("_cum_sum"))
        .sort("i")
        .drop("i")
    )
    assert_equal_data(result, expected)


def test_over_std_var(data: Data) -> None:
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "i": [0, 1, 2, 3, 4],
        "c_std0": [0.5, 0.5, 0.816496580927726, 0.816496580927726, 0.816496580927726],
        "c_std1": [0.7071067811865476, 0.7071067811865476, 1.0, 1.0, 1.0],
        "c_var0": [
            0.25,
            0.25,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
        ],
        "c_var1": [0.5, 0.5, 1.0, 1.0, 1.0],
    }

    result = (
        dataframe(data)
        .with_columns(
            c_std0=nwp.col("c").std(ddof=0).over("a"),
            c_std1=nwp.col("c").std(ddof=1).over("a"),
            c_var0=nwp.col("c").var(ddof=0).over(ncs.string()),
            c_var1=nwp.col("c").var(ddof=1).over("a"),
        )
        .sort("i")
    )
    assert_equal_data(result, expected)


# NOTE: Supporting this for pyarrow is new 🥳
def test_over_anonymous_reduction() -> None:
    df = dataframe({"a": [1, 1, 2], "b": [4, 5, 6]})
    result = df.with_columns(nwp.all().sum().over("a").name.suffix("_sum")).sort("a", "b")
    expected = {"a": [1, 1, 2], "b": [4, 5, 6], "a_sum": [2, 2, 2], "b_sum": [9, 9, 6]}
    assert_equal_data(result, expected)


def test_over_raise_len_change(data: Data) -> None:
    df = dataframe(data)
    with pytest.raises(InvalidOperationError):
        df.select(nwp.col("b").drop_nulls().over("a"))


# NOTE: Slightly different error, but same reason for raising
# (expr-ir): InvalidOperationError: `cum_sum()` is not supported in a `group_by` context
# (main): NotImplementedError: Only aggregation or literal operations are supported in grouped `over` context for PyArrow.
# https://github.com/narwhals-dev/narwhals/blob/ecde261d799a711c2e0a7acf11b108bc45035dc9/narwhals/_arrow/expr.py#L116-L118
def test_unsupported_over(data: Data) -> None:
    df = dataframe(data)
    with pytest.raises(InvalidOperationError):
        df.select(nwp.col("a").shift(1).cum_sum().over("b"))


def test_over_without_partition_by() -> None:
    df = dataframe({"a": [1, -1, 2], "i": [0, 2, 1]})
    result = (
        df.with_columns(b=nwp.col("a").abs().cum_sum().over(order_by="i"))
        .sort("i")
        .select("a", "b", "i")
    )
    expected = {"a": [1, 2, -1], "b": [1, 3, 4], "i": [0, 1, 2]}
    assert_equal_data(result, expected)


def test_aggregation_over_without_partition_by() -> None:
    df = dataframe({"a": [1, -1, 2], "i": [0, 2, 1]})
    result = (
        df.with_columns(b=nwp.col("a").diff().sum().over(order_by="i"))
        .sort("i")
        .select("a", "b", "i")
    )
    expected = {"a": [1, 2, -1], "b": [-2, -2, -2], "i": [0, 1, 2]}
    assert_equal_data(result, expected)


def test_len_over_2369() -> None:
    df = dataframe({"a": [1, 2, 4], "b": ["x", "x", "y"]})
    result = df.with_columns(a_len_per_group=nwp.len().over("b")).sort("a")
    expected = {"a": [1, 2, 4], "b": ["x", "x", "y"], "a_len_per_group": [2, 2, 1]}
    assert_equal_data(result, expected)


def test_shift_kitchen_sink(data_alt: Data) -> None:
    result = dataframe(data_alt).select(
        nwp.nth(1, 2)
        .shift(-1)
        .over(order_by=ncs.last())
        .sort(nulls_last=True)
        .fill_null(100)
        * 5
    )
    expected = {"b": [0, 5, 10, 15, 500], "c": [5, 5, 10, 45, 500]}
    assert_equal_data(result, expected)


def test_over_order_by_expr(data_alt: Data) -> None:
    df = dataframe(data_alt)
    result = df.select(
        nwp.all()
        + nwp.all().last().over(order_by=[nwp.nth(1), ncs.first()], descending=True)
    )
    expected = {"a": [6, 8, 4, 5, None], "b": [0, 1, 3, 2, 1], "c": [18, 10, 11, 10, 10]}
    assert_equal_data(result, expected)


def test_over_order_by_expr_invalid(data_alt: Data) -> None:
    df = dataframe(data_alt)
    with pytest.raises(
        InvalidOperationError,
        match=re_compile(r"only.+column.+selection.+in.+order_by.+found.+sort"),
    ):
        df.select(nwp.col("a").first().over(order_by=nwp.col("b").sort()))


def test_null_count_over() -> None:
    data = {
        "a": ["a", "b", None, None, "b", "c"],
        "b": [1, 2, 1, 5, 3, 3],
        "c": [5, 4, 3, 6, 2, 1],
    }
    expected = {
        "a": ["a", "b", None, None, "b", "c"],
        "b": [1, 2, 1, 5, 3, 3],
        "c": [5, 4, 3, 6, 2, 1],
        "first_null_count_over_b": [1, 0, 1, 1, 0, 0],
    }
    df = dataframe(data)
    result = df.with_columns(
        first_null_count_over_b=ncs.first()
        .null_count()
        .over(ncs.integer() - ncs.by_name("c"))
    )
    assert_equal_data(result, expected)


@pytest.fixture
def data_groups() -> Data:
    return {
        "a": ["a", "b", "d", "d", "b", "c"],
        "b": [1, 2, 1, 5, 3, 3],
        "c": [5, 4, 3, 6, 2, 1],
        #     ^        ^  ^  ^ = Only value in group `"c"`
        #     |        |  2 = Last (first -> descending) value in group `"b"`
        #     |        6 = Last (first -> descending) value in group `None`/"d"
        #     5 = Only value in group `"a"`
        # NOTE: Joining back is an issue for `None` group
        "i": [0, 1, 2, 3, 4, 5],
    }


@pytest.fixture
def data_groups_nulls(data_groups: Data) -> Data:
    a_d_nulls = [el if el != "d" else None for el in data_groups["a"]]
    return data_groups | {"a": a_d_nulls}


@pytest.mark.xfail(
    reason="https://github.com/narwhals-dev/narwhals/issues/3300", raises=AssertionError
)
def test_over_partition_by_nulls_order_by(data_groups_nulls: Data) -> None:
    expected = data_groups_nulls | {"result": [5, 2, 6, 6, 2, 1]}
    df = dataframe(data_groups_nulls)
    expr = nwp.col("c").first().over("a", order_by="i", descending=True)
    result = df.with_columns(result=expr).sort("i")
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("expr", "result_values"),
    [
        (
            nwp.col("c").first().over("a", order_by="i", descending=True),
            [5, 2, 6, 6, 2, 1],
        ),
        (nwp.col("c").first().over("a", order_by="i"), [5, 4, 3, 3, 4, 1]),
        (
            nwp.col("c").mean().over(ncs.integer(), order_by="i"),
            [5.0, 4.0, 3.0, 6.0, 2.0, 1.0],
        ),
        (
            nwp.col("c").min().over(ncs.first(), order_by=[ncs.first(), ncs.last()]),
            [5, 2, 3, 3, 2, 1],
        ),
    ],
)
def test_over_partition_by_order_by(
    data_groups: Data, expr: nwp.Expr, result_values: list[Any]
) -> None:
    expected = data_groups | {"result": result_values}
    df = dataframe(data_groups)
    result = df.with_columns(result=expr).sort("i")
    assert_equal_data(result, expected)


ValueColumn: TypeAlias = Literal["v1", "v2", "v3"]
OrderColumn: TypeAlias = Literal["o1", "o2", "o3", "o4", "o5"]
Agg: TypeAlias = Literal["first", "last"]

DataOrder: TypeAlias = Mapping[str, list[NonNestedLiteral]]

_AGG_EXPR_FUNCTION: Mapping[Agg, Callable[[str | Sequence[str]], nwp.Expr]] = {
    "first": first,
    "last": last,
}


@pytest.fixture(scope="module")
def data_order() -> Mapping[str, list[NonNestedLiteral]]:
    return {
        "o1": [0, 1, 2, 3],
        "o2": ["y", "y", "x", "a"],
        "o3": [None, 5, 2, 5],
        "o4": ["L", "M", "A", None],
        "o5": [1, None, None, -1],
        "v1": [12, 1, 5, 2],
        "v2": ["under", "water", "unicorn", "magic"],
        "v3": [5.9, 1.2, 22.9, 999.1],
    }


def order_case(
    columns: ValueColumn | list[ValueColumn],
    aggregation: Agg,
    /,
    order_by: OrderColumn | Sequence[OrderColumn],
    *,
    expected: NonNestedLiteral | list[NonNestedLiteral],
    descending: bool = False,
    nulls_last: bool = False,
) -> ParameterSet:
    direction = "desc" if descending else "asc"
    nulls = "nulls_last" if nulls_last else "nulls_first"
    suffix = f"_{aggregation}-{order_by}-{direction}-{nulls}"
    column_names = [columns] if not isinstance(columns, list) else columns
    expected_values = [expected] if not isinstance(expected, list) else expected
    expr = (
        _AGG_EXPR_FUNCTION[aggregation](columns)
        .over(order_by=order_by, descending=descending, nulls_last=nulls_last)
        .name.suffix(suffix)
    )
    return pytest.param(
        expr,
        {
            f"{name}{suffix}": [expect]
            for name, expect in zip_strict(column_names, expected_values)
        },
        id=f"{columns}{suffix}",
    )


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        order_case("v1", "first", order_by="o4", expected=2),
        order_case("v1", "first", order_by="o4", expected=2, descending=True),
        order_case("v1", "first", order_by="o4", expected=5, nulls_last=True),
        order_case(
            "v1", "first", order_by="o4", expected=1, descending=True, nulls_last=True
        ),
        order_case("v2", "last", order_by=["o3", "o5"], expected="magic"),
        order_case(
            "v2", "last", order_by=["o3", "o5"], expected="unicorn", descending=True
        ),
        order_case(
            "v2", "last", order_by=["o3", "o5"], expected="under", nulls_last=True
        ),
        order_case(
            "v2",
            "last",
            order_by=["o3", "o5"],
            expected="under",
            descending=True,
            nulls_last=True,
        ),
        order_case(["v3", "v2"], "last", order_by=["o2", "o5"], expected=[5.9, "under"]),
        order_case(
            ["v3", "v2"],
            "first",
            order_by=["o2", "o5"],
            expected=[1.2, "water"],
            descending=True,
        ),
        order_case(
            ["v3", "v2"],
            "first",
            order_by=["o2", "o5"],
            expected=[999.1, "magic"],
            nulls_last=True,
        ),
        order_case(
            ["v3", "v2"],
            "last",
            order_by=["o5", "o2"],
            expected=[22.9, "unicorn"],
            nulls_last=True,
            descending=True,
        ),
    ],
)
def test_over_order_by_asc_desc_nulls_first_last(
    expr: nwp.Expr, expected: Data, data_order: Mapping[str, list[NonNestedLiteral]]
) -> None:
    result = dataframe(data_order).select(expr)
    assert_equal_data(result, expected)


def test_over_partition_by_order_by_asc_desc_nulls_first_last() -> None:
    # https://github.com/pola-rs/polars/issues/24989
    data = {"a": [1, 1, 2], "b": [4, 5, 6], "c": [None, 7, 8], "i": [1, None, 2]}
    b_first = nwp.col("b").first()
    result = (
        dataframe(data)
        .with_columns(
            asc_nulls_first=b_first.over(
                "a", order_by="i", descending=False, nulls_last=False
            ),
            asc_nulls_last=b_first.over(
                "a", order_by="i", descending=False, nulls_last=True
            ),
            desc_nulls_first=b_first.over(
                "a", order_by="i", descending=True, nulls_last=False
            ),
            desc_nulls_last=b_first.over(
                "a", order_by="i", descending=True, nulls_last=True
            ),
        )
        .sort("i")
    )
    expected = {
        "a": [1, 1, 2],
        "b": [5, 4, 6],
        "c": [7.0, None, 8.0],
        "i": [None, 1, 2],
        "asc_nulls_first": [5, 5, 6],
        "asc_nulls_last": [4, 4, 6],
        "desc_nulls_first": [4, 4, 6],
        "desc_nulls_last": [5, 5, 6],
    }
    assert_equal_data(result, expected)
