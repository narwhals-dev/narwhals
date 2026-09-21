from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from datetime import timedelta
from typing import Any

import pytest

import narwhals as nw
from narwhals.exceptions import NarwhalsError
from tests.utils import POLARS_VERSION, ConstructorEager, assert_equal_data

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


def _xfail_unsupported(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    *,
    polars_min: tuple[int, ...] = (1, 0),
) -> None:
    """Mark as xfail the backends which cannot run the pivot under test.

    `polars_min` is the first polars version behaving as the test expects.
    """
    constructor_id = str(constructor_eager)
    if any(x in constructor_id for x in ("pyarrow_table", "modin")):
        reason = f"pivot is not implemented for {constructor_id}"
        request.applymarker(pytest.mark.xfail(reason=reason))
    if "polars" in constructor_id and polars_min > POLARS_VERSION:  # pragma: no cover
        version = ".".join(map(str, polars_min))
        reason = f"this pivot behaviour requires polars>={version}"
        request.applymarker(pytest.mark.xfail(reason=reason))


@pytest.mark.parametrize(
    ("agg_func", "expected"),
    [
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
    ],
)
@pytest.mark.parametrize(("on", "index"), [("col", "ix"), (["col"], ["ix"])])
def test_pivot(
    constructor_eager: ConstructorEager,
    agg_func: str,
    expected: dict[str, list[Any]],
    on: str | list[str],
    index: str | list[str],
    request: pytest.FixtureRequest,
) -> None:
    _xfail_unsupported(constructor_eager, request)

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
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    data_: Any,
    context: Any,
) -> None:
    _xfail_unsupported(constructor_eager, request)

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
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    sort_columns: Any,
    expected: list[str],
) -> None:
    _xfail_unsupported(constructor_eager, request)

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
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    kwargs: Any,
    expected: list[str],
) -> None:
    _xfail_unsupported(constructor_eager, request)

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
    _xfail_unsupported(constructor_eager, request)
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


def test_pivot_on_columns_str_raises(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data_no_dups), eager_only=True)
    with pytest.raises(TypeError, match="on_columns"):
        df.pivot("col", on_columns="a", index="ix", values="foo")


def test_pivot_on_columns_multiple_on_raises(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(NotImplementedError, match="single column"):
        df.pivot(["col", "col_b"], on_columns=["a"], index="ix", values="foo")


@pytest.mark.skipif(
    not ((1, 0) <= POLARS_VERSION < (1, 36)),
    reason="`pivot` needs polars>=1, and `on_columns` is native from 1.36",
)
def test_pivot_on_columns_polars_too_old() -> None:  # pragma: no cover
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame(data_no_dups), eager_only=True)
    with pytest.raises(NotImplementedError, match=r"polars>=1\.36\.0"):
        df.pivot("col", on_columns=["a"], index="ix", values="foo")


@pytest.mark.parametrize(
    ("on_columns", "values", "expected"),
    [
        # Order follows `on_columns`, not discovery nor sorting.
        (["b", "a"], ["foo"], ["ix", "b", "a"]),
        (["a", "b"], ["foo"], ["ix", "a", "b"]),
        (["b", "a"], ["foo", "bar"], ["ix", "foo_b", "foo_a", "bar_b", "bar_a"]),
        # Subset: values present in the data but not listed are dropped.
        (["a"], ["foo", "bar"], ["ix", "foo_a", "bar_a"]),
        # Superset: unknown values still produce a column.
        (["a", "z", "b"], ["foo"], ["ix", "a", "z", "b"]),
        ([], ["foo"], ["ix"]),
    ],
)
def test_pivot_on_columns_names_out(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    on_columns: list[str],
    values: list[str],
    expected: list[str],
) -> None:
    _xfail_unsupported(constructor_eager, request, polars_min=(1, 36))
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.pivot(
        "col",
        on_columns=on_columns,
        index="ix",
        values=values,
        aggregate_function="min",
        sort_columns=True,
    )
    assert result.collect_schema().names() == expected


@pytest.mark.parametrize(
    ("agg_func", "expected"),
    [
        ("min", {"ix": [1, 2], "b": [7, 1], "a": [0, 2], "z": [None, None]}),
        ("max", {"ix": [1, 2], "b": [7, 1], "a": [1, 2], "z": [None, None]}),
        ("first", {"ix": [1, 2], "b": [7, 1], "a": [0, 2], "z": [None, None]}),
        ("last", {"ix": [1, 2], "b": [7, 1], "a": [1, 2], "z": [None, None]}),
        ("mean", {"ix": [1, 2], "b": [7.0, 1.0], "a": [0.5, 2.0], "z": [None, None]}),
        ("median", {"ix": [1, 2], "b": [7.0, 1.0], "a": [0.5, 2.0], "z": [None, None]}),
        # `sum` and `len` of an empty group are 0, not null.
        ("sum", {"ix": [1, 2], "b": [7, 1], "a": [1, 4], "z": [0, 0]}),
        ("len", {"ix": [1, 2], "b": [1, 1], "a": [2, 2], "z": [0, 0]}),
    ],
)
def test_pivot_on_columns_agg(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    agg_func: str,
    expected: dict[str, list[Any]],
) -> None:
    _xfail_unsupported(constructor_eager, request, polars_min=(1, 36))
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.pivot(
        "col",
        on_columns=["b", "a", "z"],
        index="ix",
        values="foo",
        aggregate_function=agg_func,  # type: ignore[arg-type]
    )
    assert_equal_data(result, expected)


def test_pivot_on_columns_no_agg(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    _xfail_unsupported(constructor_eager, request, polars_min=(1, 36))
    df = nw.from_native(constructor_eager(data_no_dups), eager_only=True)
    result = df.pivot("col", on_columns=["b", "z"], index="ix", values="foo")
    expected = {"ix": [1, 2], "b": [2, 4], "z": [None, None]}
    assert_equal_data(result, expected)
    # Duplicates in the selected `on` values still raise without an aggregation.
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises((ValueError, NarwhalsError)):
        df.pivot("col", on_columns=["a"], index="ix", values="foo")


def test_pivot_on_columns_no_index(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    _xfail_unsupported(constructor_eager, request, polars_min=(1, 36))
    df = nw.from_native(constructor_eager(data_no_dups), eager_only=True)
    result = df.pivot("col", on_columns=["a"], values="foo").sort("ix", "bar")
    expected = {
        "ix": [1, 1, 2, 2],
        "bar": ["x", "y", "w", "z"],
        "a": [1.0, None, None, 3.0],
    }
    assert_equal_data(result, expected)


def test_pivot_on_columns_series(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    _xfail_unsupported(constructor_eager, request, polars_min=(1, 36))
    df = nw.from_native(constructor_eager(data), eager_only=True)
    on_columns = nw.from_native(constructor_eager({"c": ["b", "z"]}), eager_only=True)[
        "c"
    ]
    result = df.pivot(
        "col", on_columns=on_columns, index="ix", values="foo", aggregate_function="max"
    )
    expected = {"ix": [1, 2], "b": [7, 1], "z": [None, None]}
    assert_equal_data(result, expected)


def test_pivot_on_columns_keeps_index_rows(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    # An index value whose `on` values are all excluded still yields a (null) row.
    _xfail_unsupported(constructor_eager, request, polars_min=(1, 36))
    data_ = {"ix": [3, 1, 1, 2], "col": ["c", "a", "b", "b"], "foo": [9, 1, 2, 4]}
    df = nw.from_native(constructor_eager(data_), eager_only=True)
    result = df.pivot("col", on_columns=["a", "z"], index="ix", values="foo").sort("ix")
    expected = {"ix": [1, 2, 3], "a": [1, None, None], "z": [None, None, None]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("agg_func", "expected", "polars_min"),
    [
        ("sum", {"ix": [1, 2], "a": [4, 0], "b": [0, 2]}, (1, 32)),
        ("len", {"ix": [1, 2], "a": [2, 0], "b": [0, 1]}, (1, 32)),
        ("min", {"ix": [1, 2], "a": [1, None], "b": [None, 2]}, (1, 0)),
    ],
)
def test_pivot_empty_group(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    agg_func: str,
    expected: dict[str, list[Any]],
    polars_min: tuple[int, ...],
) -> None:
    _xfail_unsupported(constructor_eager, request, polars_min=polars_min)
    data_ = {"ix": [1, 2, 1], "col": ["a", "b", "a"], "foo": [1, 2, 3]}
    df = nw.from_native(constructor_eager(data_), eager_only=True)
    result = df.pivot(
        "col",
        index="ix",
        values="foo",
        aggregate_function=agg_func,  # type: ignore[arg-type]
        sort_columns=True,
    )
    assert_equal_data(result, expected)


def test_pivot_on_columns_null_dtype(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    _xfail_unsupported(constructor_eager, request, polars_min=(1, 36))
    data_ = {"ix": [1, 2], "col": ["a", "b"], "foo": [1, 2], "bar": ["x", "y"]}
    df = nw.from_native(constructor_eager(data_), eager_only=True)
    schema = df.pivot(
        "col",
        on_columns=["a", "z"],
        index="ix",
        values=["foo", "bar"],
        aggregate_function="min",
    ).schema
    # `foo_a` and `bar_a` are present but hold a null themselves, so the absent
    # columns must have exactly their dtypes on every backend.
    assert schema["foo_z"] == schema["foo_a"]
    assert schema["bar_z"] == schema["bar_a"]
    assert schema["bar_z"] == nw.String


def test_pivot_on_columns_multi_index(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    _xfail_unsupported(constructor_eager, request, polars_min=(1, 36))
    data_ = {"ix": [1, 2], "iy": ["p", "q"], "col": ["a", "b"], "foo": [1, 2]}
    df = nw.from_native(constructor_eager(data_), eager_only=True)
    result = df.pivot(
        "col", on_columns=["a", "z"], index=["ix", "iy"], values="foo"
    ).sort("ix")
    expected = {"ix": [1, 2], "iy": ["p", "q"], "a": [1, None], "z": [None, None]}
    assert_equal_data(result, expected)


def test_pivot_on_columns_no_index_absent(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    # `index=None` infers every remaining column, so the default is a multi-level one.
    _xfail_unsupported(constructor_eager, request, polars_min=(1, 36))
    df = nw.from_native(constructor_eager(data_no_dups), eager_only=True)
    result = df.pivot("col", on_columns=["a", "z"], values="foo").sort("ix", "bar")
    expected = {
        "ix": [1, 1, 2, 2],
        "bar": ["x", "y", "w", "z"],
        "a": [1.0, None, None, 3.0],
        "z": [None, None, None, None],
    }
    assert_equal_data(result, expected)


def test_pivot_multi_on_unobserved(constructor_eager: ConstructorEager) -> None:
    if "pandas" not in str(constructor_eager):
        pytest.skip("pandas-like backends only")

    data_ = {"ix": [1, 2], "col": ["a", "b"], "col_b": ["x", "y"], "foo": [1, 2]}
    df = nw.from_native(constructor_eager(data_), eager_only=True)
    with pytest.raises((KeyError, NarwhalsError)):
        df.pivot(["col", "col_b"], index="ix", values="foo", aggregate_function="min")


def test_pivot_empty_group_non_numeric(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    # Nanoseconds because `pandas<2.2` pivots a coarser unit to its int64 sentinel
    # rather than to a null, which no fill can reach.
    _xfail_unsupported(constructor_eager, request, polars_min=(1, 32))
    data_ = {"ix": [1, 2], "col": ["a", "b"], "foo": [1_000, 2_000]}
    df = nw.from_native(constructor_eager(data_), eager_only=True).with_columns(
        nw.col("foo").cast(nw.Duration("ns"))
    )
    result = df.pivot(
        "col", index="ix", values="foo", aggregate_function="sum", sort_columns=True
    )
    assert result.schema["a"] == nw.Duration("ns")
    assert result.schema["b"] == nw.Duration("ns")
    expected = {
        "ix": [1, 2],
        "a": [timedelta(microseconds=1), timedelta(0)],
        "b": [timedelta(0), timedelta(microseconds=2)],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize("agg_func", ["len", "min"])
def test_pivot_on_columns_empty_frame(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest, agg_func: str
) -> None:
    # With no rows there is no pivoted column to take the dtype from, and `on_columns`
    # promises a schema that does not depend on the data.
    _xfail_unsupported(constructor_eager, request, polars_min=(1, 36))
    cases: list[dict[str, list[Any]]] = [
        {"ix": [], "col": [], "bar": []},
        {"ix": [1], "col": ["b"], "bar": ["x"]},
    ]
    schemas = []
    for rows in cases:
        df = nw.from_native(constructor_eager(rows), eager_only=True).with_columns(
            nw.col("ix").cast(nw.Int64), nw.col("col", "bar").cast(nw.String)
        )
        schemas.append(
            df.pivot(
                "col",
                on_columns=["a"],
                index="ix",
                values="bar",
                aggregate_function=agg_func,  # type: ignore[arg-type]
            ).schema
        )
    assert schemas[0] == schemas[1]


def test_pivot_on_columns_all_null_values(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    # An all-null `values` column pivots to no column at all on pandas, reaching the
    # same dtype fallback with rows present.
    _xfail_unsupported(constructor_eager, request, polars_min=(1, 36))
    data_: dict[str, list[Any]] = {"ix": [1, 2], "col": ["a", "b"], "foo": [None, None]}
    df = nw.from_native(constructor_eager(data_), eager_only=True)
    result = df.pivot("col", on_columns=["a", "z"], index="ix", values="foo").sort("ix")
    expected = {"ix": [1, 2], "a": [None, None], "z": [None, None]}
    assert_equal_data(result, expected)
    assert result.schema["z"] == result.schema["a"]


def test_pivot_on_columns_excluded_duplicates(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    # Duplicates confined to an `on` value nobody asked for must not raise.
    _xfail_unsupported(constructor_eager, request, polars_min=(1, 36))
    data_ = {"ix": [1, 1, 2], "col": ["b", "b", "a"], "foo": [1, 2, 3]}
    df = nw.from_native(constructor_eager(data_), eager_only=True)
    result = df.pivot("col", on_columns=["a"], index="ix", values="foo").sort("ix")
    assert_equal_data(result, {"ix": [1, 2], "a": [None, 3]})
