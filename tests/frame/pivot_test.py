from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.exceptions import NarwhalsError
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import LazyPivotAgg

data = {
    "ix": [1, 2, 1, 1, 2, 2],
    "iy": [1, 2, 2, 1, 2, 1],
    "col": ["b", "b", "a", "a", "a", "a"],
    "col_b": ["x", "y", "x", "y", "x", "y"],
    "foo": [7, 1, 0, 1, 2, 2],
    "bar": [9, 4, 0, 2, 0, 0],
}

data_no_dups = {
    "ix": [1, 1, 2, 2],
    "col": ["a", "b", "a", "b"],
    "foo": [1, 2, 3, 4],
    "bar": ["x", "y", "z", "w"],
}

data_missing_combination = {
    "name": ["Cady", "Cady", "Karen"],
    "subject": ["maths", "physics", "maths"],
    "score": [98, 99, 61],
}

PIVOT_MISSING_COMBINATION_CASES = [
    ("sum", {"name": ["Cady", "Karen"], "maths": [98, 61], "physics": [99, 0]}),
    ("len", {"name": ["Cady", "Karen"], "maths": [1, 1], "physics": [1, 0]}),
    ("mean", {"name": ["Cady", "Karen"], "maths": [98.0, 61.0], "physics": [99.0, None]}),
]


PIVOT_CASES = [
    (
        "min",
        {
            "ix": [1, 2],
            "foo_a": [0, 2],
            "foo_b": [7, 1],
            "bar_a": [0, 0],
            "bar_b": [9, 4],
        },
    ),
    (
        "max",
        {
            "ix": [1, 2],
            "foo_a": [1, 2],
            "foo_b": [7, 1],
            "bar_a": [2, 0],
            "bar_b": [9, 4],
        },
    ),
    (
        "first",
        {
            "ix": [1, 2],
            "foo_a": [0, 2],
            "foo_b": [7, 1],
            "bar_a": [0, 0],
            "bar_b": [9, 4],
        },
    ),
    (
        "last",
        {
            "ix": [1, 2],
            "foo_a": [1, 2],
            "foo_b": [7, 1],
            "bar_a": [2, 0],
            "bar_b": [9, 4],
        },
    ),
    (
        "sum",
        {
            "ix": [1, 2],
            "foo_a": [1, 4],
            "foo_b": [7, 1],
            "bar_a": [2, 0],
            "bar_b": [9, 4],
        },
    ),
    (
        "mean",
        {
            "ix": [1, 2],
            "foo_a": [0.5, 2.0],
            "foo_b": [7.0, 1.0],
            "bar_a": [1.0, 0.0],
            "bar_b": [9.0, 4.0],
        },
    ),
    (
        "median",
        {
            "ix": [1, 2],
            "foo_a": [0.5, 2.0],
            "foo_b": [7.0, 1.0],
            "bar_a": [1.0, 0.0],
            "bar_b": [9.0, 4.0],
        },
    ),
    (
        "len",
        {
            "ix": [1, 2],
            "foo_a": [2, 2],
            "foo_b": [1, 1],
            "bar_a": [2, 2],
            "bar_b": [1, 1],
        },
    ),
]


def make_lazy_frame(data_: Any, constructor: Constructor) -> nw.LazyFrame[Any]:
    frame = nw.from_native(constructor(data_))
    if isinstance(frame, nw.LazyFrame):
        if frame.implementation is nw.Implementation.POLARS and POLARS_VERSION < (1, 43):
            pytest.skip("Polars LazyFrame.pivot")
        return frame
    msg = "LazyFrame.pivot"
    raise pytest.skip.Exception(msg)


@pytest.mark.parametrize(("agg_func", "expected"), PIVOT_CASES)
@pytest.mark.parametrize(("on", "index"), [("col", "ix"), (["col"], ["ix"])])
def test_pivot(
    constructor_eager: ConstructorEager,
    agg_func: str,
    expected: dict[str, list[Any]],
    on: str | list[str],
    index: str | list[str],
    request: pytest.FixtureRequest,
) -> None:
    if any(x in str(constructor_eager) for x in ("pyarrow_table", "modin")):
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 0):
        # not implemented
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.pivot(
        on=on,
        index=index,
        values=["foo", "bar"],
        aggregate_function=agg_func,  # type: ignore[arg-type]
        sort_columns=True,
    )

    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("data_", "context"),
    [
        (data_no_dups, does_not_raise()),
        (data, pytest.raises((ValueError, NarwhalsError))),
    ],
)
def test_pivot_no_agg(
    request: Any, constructor_eager: ConstructorEager, data_: Any, context: Any
) -> None:
    if any(x in str(constructor_eager) for x in ("pyarrow_table", "modin")):
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 0):
        # not implemented
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data_), eager_only=True)
    with context:
        df.pivot("col", index="ix", aggregate_function=None)


@pytest.mark.parametrize(
    ("sort_columns", "expected"),
    [
        (True, ["ix", "foo_a", "foo_b", "bar_a", "bar_b"]),
        (False, ["ix", "foo_b", "foo_a", "bar_b", "bar_a"]),
    ],
)
def test_pivot_sort_columns(
    request: Any,
    constructor_eager: ConstructorEager,
    sort_columns: Any,
    expected: list[str],
) -> None:
    if any(x in str(constructor_eager) for x in ("pyarrow_table", "modin")):
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 0):
        # not implemented
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.pivot(
        on="col",
        index="ix",
        values=["foo", "bar"],
        aggregate_function="sum",
        sort_columns=sort_columns,
    )
    assert result.columns == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"on": ["col"], "values": ["foo"]}, ["ix", "b", "a"]),
        (
            {"on": ["col"], "values": ["foo", "bar"]},
            ["ix", "foo_b", "foo_a", "bar_b", "bar_a"],
        ),
        (
            {"on": ["col", "col_b"], "values": ["foo"]},
            ["ix", '{"b","x"}', '{"b","y"}', '{"a","x"}', '{"a","y"}'],
        ),
        (
            {"on": ["col", "col_b"], "values": ["foo", "bar"]},
            [
                "ix",
                'foo_{"b","x"}',
                'foo_{"b","y"}',
                'foo_{"a","x"}',
                'foo_{"a","y"}',
                'bar_{"b","x"}',
                'bar_{"b","y"}',
                'bar_{"a","x"}',
                'bar_{"a","y"}',
            ],
        ),
    ],
)
def test_pivot_names_out(
    request: Any, constructor_eager: ConstructorEager, kwargs: Any, expected: list[str]
) -> None:
    if any(x in str(constructor_eager) for x in ("pyarrow_table", "modin")):
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 0):
        # not implemented
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)

    result = (
        df.pivot(aggregate_function="min", index="ix", **kwargs).collect_schema().names()
    )
    assert result == expected


def test_pivot_no_index_no_values(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data_no_dups), eager_only=True)
    with pytest.raises(ValueError, match="At least one of `values` and `index` must"):
        df.pivot(on="col")


def test_pivot_no_index(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor_eager) for x in ("pyarrow_table", "modin")):
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 0):
        # not implemented
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data_no_dups), eager_only=True)
    with pytest.warns(UserWarning, match="has no effect"):
        result = df.pivot(on="col", values="foo", maintain_order=True).sort("ix", "bar")
    expected = {
        "ix": [1, 1, 2, 2],
        "bar": ["x", "y", "w", "z"],
        "a": [1.0, None, None, 3.0],
        "b": [None, 2.0, 4.0, None],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("agg_func", "expected"), PIVOT_CASES)
@pytest.mark.parametrize("index", ["ix", ["ix"]])
def test_pivot_lazy(
    constructor: Constructor,
    agg_func: LazyPivotAgg,
    expected: dict[str, list[Any]],
    index: str | list[str],
) -> None:
    df = make_lazy_frame(data, constructor)
    result = (
        df.pivot(
            "col",
            on_columns=["a", "b"],
            index=index,
            values=["foo", "bar"],
            aggregate_function=agg_func,
            maintain_order=df.implementation is nw.Implementation.POLARS,
        )
        .sort("ix")
        .collect()
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("agg_func", "expected"), PIVOT_MISSING_COMBINATION_CASES)
def test_pivot_lazy_missing_combination(
    constructor: Constructor, agg_func: LazyPivotAgg, expected: dict[str, list[Any]]
) -> None:
    df = make_lazy_frame(data_missing_combination, constructor)
    result = (
        df.pivot(
            "subject",
            on_columns=["maths", "physics"],
            index="name",
            values="score",
            aggregate_function=agg_func,
            maintain_order=df.implementation is nw.Implementation.POLARS,
        )
        .sort("name")
        .collect()
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("data_", "context"),
    [
        (data_no_dups, does_not_raise()),
        (data, pytest.raises((ValueError, NarwhalsError))),
    ],
)
def test_pivot_lazy_no_agg(constructor: Constructor, data_: Any, context: Any) -> None:
    df = make_lazy_frame(data_, constructor)
    if df.implementation is not nw.Implementation.POLARS:
        context = pytest.raises(
            NotImplementedError, match="cannot validate that each group"
        )
    with context:
        df.pivot("col", ["a", "b"], index="ix").collect()


def test_pivot_lazy_no_index_no_values(constructor: Constructor) -> None:
    df = make_lazy_frame(data_no_dups, constructor)
    with pytest.raises(ValueError, match="At least one of `values` and `index` must"):
        df.pivot("col", ["a", "b"])


def test_pivot_lazy_no_index(constructor: Constructor) -> None:
    df = make_lazy_frame(data_no_dups, constructor)
    aggregate_function: LazyPivotAgg = (
        "min" if df.implementation is nw.Implementation.SQLFRAME else "first"
    )
    result = (
        df.pivot(
            "col",
            ["a", "b"],
            values="foo",
            aggregate_function=aggregate_function,
            maintain_order=df.implementation is nw.Implementation.POLARS,
        )
        .sort("ix", "bar")
        .collect()
    )
    expected = {
        "ix": [1, 1, 2, 2],
        "bar": ["x", "y", "w", "z"],
        "a": [1, None, None, 3],
        "b": [None, 2, 4, None],
    }
    assert_equal_data(result, expected)


def test_pivot_lazy_maintain_order(constructor: Constructor) -> None:
    df = make_lazy_frame(data, constructor)
    if df.implementation is nw.Implementation.POLARS:
        pytest.skip("Polars supports maintaining row order during a pivot")
    with pytest.raises(NotImplementedError, match="maintaining row order"):
        df.pivot(
            "col",
            ["a", "b"],
            index="ix",
            values="foo",
            aggregate_function="sum",
            maintain_order=True,
        )
