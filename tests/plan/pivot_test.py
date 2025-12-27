from __future__ import annotations

# ruff: noqa: FBT001
from typing import TYPE_CHECKING

import pytest

import narwhals._plan.selectors as ncs
from narwhals.exceptions import NarwhalsError
from tests.plan.utils import assert_equal_data, dataframe, re_compile

if TYPE_CHECKING:
    from narwhals.typing import PivotAgg
    from tests.conftest import Data

# TODO @dangotbanned: Add more values for `median`, like in
# https://github.com/narwhals-dev/narwhals/blob/b3d8c7349bbf7ecb7f11ea590c334e12d5c1d43e/tests/plan/list_agg_test.py#L24-L34
XFAIL_PYARROW_MEDIAN = pytest.mark.xfail(
    reason="Tried to use `'approximate_median'` but groups are too small",
    raises=AssertionError,
)


@pytest.fixture(scope="module")
def scores() -> Data:
    """Dataset 1 `pl.DataFrame.pivot` docstring."""
    return {
        "name": ["Cady", "Cady", "Karen", "Karen"],
        "subject": ["maths", "physics", "maths", "physics"],
        "test_1": [98, 99, 61, 58],
        "test_2": [100, 100, 60, 60],
    }


@pytest.fixture(scope="module")
def tanh_mean() -> Data:
    """Dataset 3 `pl.DataFrame.pivot` docstring."""
    return {
        "col1": ["a", "a", "a", "b", "b", "b"],
        "col2": ["x", "x", "x", "x", "y", "y"],
        "col3": [6, 7, 3, 2, 5, 7],
    }


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "idx_1": [1, 2, 1, 1, 2, 2],
        "idx_2": [1, 2, 2, 1, 2, 1],
        "on_lower": ["b", "b", "a", "a", "a", "a"],
        "on_upper": ["X", "Y", "Y", "Y", "X", "Y"],
        "foo": [7, 1, 0, 1, 2, 2],
        "bar": [9, 4, 0, 2, 0, 0],
    }


@pytest.fixture(scope="module")
def data_no_dups() -> Data:
    return {
        "idx_1": [1, 1, 2, 2],
        "on_lower": ["a", "b", "a", "b"],
        "foo": [1, 2, 3, 4],
        "bar": ["x", "y", "z", "w"],
    }


@pytest.fixture(scope="module")
def data_no_dups_unordered(data_no_dups: Data) -> Data:
    """Variant of `data_no_dups` to support tests without needing `aggregate_function`.

    - `"on_lower"` has an order to test `sort_columns=True`
    - `"on_upper"` is added for `on: list[str]` name generation
    """
    return data_no_dups | {
        "on_lower": ["b", "a", "b", "a"],
        "on_upper": ["X", "X", "Y", "Y"],
    }


def assert_names_match_polars(
    input_data: Data,
    on: list[str],
    index: str | list[str],
    values: list[str],
    aggregate_function: PivotAgg | None = None,
    *,
    result_columns: list[str],
) -> None:
    """Ensure the complex renaming cases match upstream."""
    pytest.importorskip("polars")
    import polars as pl

    df = pl.DataFrame(input_data)

    pl_result = df.pivot(
        on=on, values=values, index=index, aggregate_function=aggregate_function
    )
    assert result_columns == pl_result.columns


@pytest.mark.parametrize(
    ("agg_func", "expected"),
    [
        (
            "min",
            {
                "idx_1": [1, 2],
                "foo_a": [0, 2],
                "foo_b": [7, 1],
                "bar_a": [0, 0],
                "bar_b": [9, 4],
            },
        ),
        (
            "max",
            {
                "idx_1": [1, 2],
                "foo_a": [1, 2],
                "foo_b": [7, 1],
                "bar_a": [2, 0],
                "bar_b": [9, 4],
            },
        ),
        (
            "first",
            {
                "idx_1": [1, 2],
                "foo_a": [0, 2],
                "foo_b": [7, 1],
                "bar_a": [0, 0],
                "bar_b": [9, 4],
            },
        ),
        (
            "last",
            {
                "idx_1": [1, 2],
                "foo_a": [1, 2],
                "foo_b": [7, 1],
                "bar_a": [2, 0],
                "bar_b": [9, 4],
            },
        ),
        (
            "sum",
            {
                "idx_1": [1, 2],
                "foo_a": [1, 4],
                "foo_b": [7, 1],
                "bar_a": [2, 0],
                "bar_b": [9, 4],
            },
        ),
        (
            "mean",
            {
                "idx_1": [1, 2],
                "foo_a": [0.5, 2.0],
                "foo_b": [7.0, 1.0],
                "bar_a": [1.0, 0.0],
                "bar_b": [9.0, 4.0],
            },
        ),
        pytest.param(
            "median",
            {
                "idx_1": [1, 2],
                "foo_a": [0.5, 2.0],
                "foo_b": [7.0, 1.0],
                "bar_a": [1.0, 0.0],
                "bar_b": [9.0, 4.0],
            },
            marks=XFAIL_PYARROW_MEDIAN,
        ),
        (
            "len",
            {
                "idx_1": [1, 2],
                "foo_a": [2, 2],
                "foo_b": [1, 1],
                "bar_a": [2, 2],
                "bar_b": [1, 1],
            },
        ),
    ],
)
@pytest.mark.parametrize(
    ("on", "index"), [("on_lower", "idx_1"), (["on_lower"], ["idx_1"])]
)
def test_pivot_agg(
    data: Data,
    on: str | list[str],
    index: str | list[str],
    agg_func: PivotAgg,
    expected: Data,
) -> None:
    df = dataframe(data)
    result = df.pivot(
        on,
        index=index,
        values=["foo", "bar"],
        aggregate_function=agg_func,
        sort_columns=True,
    )

    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("sort_columns", "expected"),
    [
        (True, ["idx_1", "foo_a", "foo_b", "bar_a", "bar_b"]),
        (False, ["idx_1", "foo_b", "foo_a", "bar_b", "bar_a"]),
    ],
)
def test_pivot_sort_columns(
    data_no_dups_unordered: Data, sort_columns: bool, expected: list[str]
) -> None:
    df = dataframe(data_no_dups_unordered)
    values = ["foo", "bar"]
    result = df.pivot("on_lower", index="idx_1", values=values, sort_columns=sort_columns)
    assert result.columns == expected


@pytest.mark.parametrize(
    ("on", "values", "expected"),
    [
        (
            ["on_lower", "on_upper"],
            ["foo"],
            ["idx_1", '{"b","X"}', '{"a","X"}', '{"b","Y"}', '{"a","Y"}'],
        ),
        (
            ["on_lower", "on_upper"],
            ["foo", "bar"],
            [
                "idx_1",
                'foo_{"b","X"}',
                'foo_{"a","X"}',
                'foo_{"b","Y"}',
                'foo_{"a","Y"}',
                'bar_{"b","X"}',
                'bar_{"a","X"}',
                'bar_{"b","Y"}',
                'bar_{"a","Y"}',
            ],
        ),
    ],
    ids=["single-values", "multiple-values"],
)
def test_pivot_on_multiple_names(
    data_no_dups_unordered: Data, on: list[str], values: list[str], expected: list[str]
) -> None:
    index = "idx_1"
    data_ = data_no_dups_unordered
    result = dataframe(data_).pivot(on, values=values, index=index)
    assert result.columns == expected
    assert_names_match_polars(data_, on, index, values, result_columns=result.columns)


@pytest.mark.parametrize(
    ("on", "values", "expected"),
    [
        (
            ["on_lower", "on_upper"],
            ["foo"],
            ["idx_1", '{"b","X"}', '{"b","Y"}', '{"a","Y"}', '{"a","X"}'],
        ),
        (
            ["on_lower", "on_upper"],
            ["foo", "bar"],
            [
                "idx_1",
                'foo_{"b","X"}',
                'foo_{"b","Y"}',
                'foo_{"a","Y"}',
                'foo_{"a","X"}',
                'bar_{"b","X"}',
                'bar_{"b","Y"}',
                'bar_{"a","Y"}',
                'bar_{"a","X"}',
            ],
        ),
    ],
    ids=["single-values", "multiple-values"],
)
def test_pivot_on_multiple_names_agg(
    data: Data, on: list[str], values: list[str], expected: list[str]
) -> None:
    index = "idx_1"
    df = dataframe(data)
    result = df.pivot(on, values=values, aggregate_function="min", index=index)
    assert result.columns == expected
    assert_names_match_polars(
        data, on, index, values, "min", result_columns=result.columns
    )


def test_pivot_no_agg_duplicated(data: Data) -> None:
    df = dataframe(data)
    with pytest.raises((ValueError, NarwhalsError)):
        df.pivot("on_lower", index="idx_1")


def test_pivot_no_agg_no_duplicates(data_no_dups: Data) -> None:
    result = dataframe(data_no_dups).pivot("on_lower", index="idx_1")
    expected = {
        "idx_1": [1, 2],
        "foo_a": [1, 3],
        "foo_b": [2, 4],
        "bar_a": ["x", "z"],
        "bar_b": ["y", "w"],
    }
    assert_equal_data(result, expected)


def test_pivot_no_index_no_values(data_no_dups: Data) -> None:
    df = dataframe(data_no_dups)
    with pytest.raises(
        ValueError, match=re_compile(r"at least one of.+values.+index.+must")
    ):
        df.pivot("on_lower")


def test_pivot_implicit_index(data_no_dups: Data) -> None:
    expected = {
        "idx_1": [1, 1, 2, 2],
        "bar": ["x", "y", "w", "z"],
        "a": [1.0, None, None, 3.0],
        "b": [None, 2.0, 4.0, None],
    }
    result = (
        dataframe(data_no_dups).pivot("on_lower", values="foo").sort(ncs.by_index(0, 1))
    )
    assert_equal_data(result, expected)


def test_pivot_test_scores_1(scores: Data) -> None:
    df = dataframe(scores)
    expected = {"name": ["Cady", "Karen"], "maths": [98, 61], "physics": [99, 58]}
    result = df.pivot("subject", index="name", values="test_1")
    assert_equal_data(result, expected)
    result = df.pivot(
        "subject", on_columns=["maths", "physics"], index="name", values="test_1"
    )
    assert_equal_data(result, expected)


def test_pivot_test_scores_2(scores: Data) -> None:
    df = dataframe(scores)
    expected = {
        "name": ["Cady", "Karen"],
        "test_1_maths": [98, 61],
        "test_1_physics": [99, 58],
        "test_2_maths": [100, 60],
        "test_2_physics": [100, 60],
    }
    result = df.pivot("subject", values=["test_1", "test_2"])
    assert_equal_data(result, expected)


# TODO @dangotbanned: Consider fixing this?
# The `pandas` impl on `main` has the same issue
@pytest.mark.parametrize(
    ("aggregate_function", "expected"),
    [
        ("first", {"x": [6, 2], "y": [None, 5]}),
        pytest.param(
            "sum",
            {"x": [16, 2], "y": [0, 12]},
            marks=pytest.mark.xfail(
                reason="`sum` is special-cased in `polars` to always return 0",
                raises=AssertionError,
            ),
        ),
    ],
    ids=["first-null", "sum-zero"],
)
def test_pivot_test_tanh_mean(
    tanh_mean: Data, aggregate_function: PivotAgg, expected: Data
) -> None:
    # NOTE: Docstring example uses `pl.element().tanh().mean()`
    # Here though, we're just reusing the dataset as it exposes an edge case
    df = dataframe(tanh_mean)
    expected = {"col1": ["a", "b"], **expected}
    result = df.pivot(
        "col2", index="col1", values="col3", aggregate_function=aggregate_function
    )
    assert_equal_data(result, expected)
