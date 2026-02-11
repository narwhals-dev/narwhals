from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from narwhals.exceptions import ColumnNotFoundError, InvalidOperationError
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}


@pytest.mark.parametrize(
    ("predicates", "expected"),
    [
        ((nw.col("a") > 1,), {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}),
        ((nw.col("a") > 1, nw.col("z") < 9.0), {"a": [3], "b": [4], "z": [8.0]}),
    ],
)
def test_filter_with_expr_predicates(
    constructor: Constructor,
    predicates: tuple[nw.Expr, ...],
    expected: dict[str, list[Any]],
) -> None:
    df = nw.from_native(constructor(data))
    result = df.filter(*predicates)
    assert_equal_data(result, expected)


def test_filter_with_series_predicates(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.filter(df["a"] > 1)
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    assert_equal_data(result, expected)

    result = df.filter(df["a"] > 1, df["b"] < 6)
    expected = {"a": [3], "b": [4], "z": [8.0]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("predicates", "expected"),
    [
        (([False, True, True],), {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}),
        (([True, True, False], [False, True, True]), {"a": [3], "b": [4], "z": [8.0]}),
    ],
)
def test_filter_with_boolean_list_predicates_eager(
    constructor_eager: ConstructorEager,
    predicates: tuple[list[bool], ...],
    expected: dict[str, list[Any]],
) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.filter(*predicates)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    "predicates",
    [
        ([False, True, True],),
        ([True, True, False], [False, True, True]),
        (nw.col("a") > 1, [False, True, True]),
        ([True, True, False], nw.col("z") < 9.0),
    ],
)
def test_filter_with_boolean_list_predicates_lazy(
    constructor: Constructor, predicates: tuple[list[bool] | nw.Expr, ...]
) -> None:
    df = nw.from_native(constructor(data)).lazy()
    with pytest.raises(TypeError, match="not supported with Python boolean masks"):
        df.filter(*predicates)  # type: ignore[arg-type]


def test_filter_raise_on_agg_predicate(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    with pytest.raises(InvalidOperationError):
        df.filter(nw.col("a").max() > 2).lazy().collect()


def test_filter_raise_on_shape_mismatch(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    with pytest.raises((InvalidOperationError, NotImplementedError)):
        df.filter(nw.col("b").unique() > 2).lazy().collect()


def test_filter_with_constrains_only(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result_scalar = df.filter(a=3)
    expected_scalar = {"a": [3], "b": [4], "z": [8.0]}

    assert_equal_data(result_scalar, expected_scalar)

    result_expr = df.filter(a=nw.col("b") // 3)
    expected_expr = {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}

    assert_equal_data(result_expr, expected_expr)


def test_filter_missing_column(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    constructor_id = str(request.node.callspec.id)
    if any(id_ == constructor_id for id_ in ("sqlframe", "pyspark[connect]", "ibis")):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    if "polars" in str(constructor):
        msg = r"unable to find column \"c\"; valid columns: \[\"a\", \"b\"\, \"z\"\]"
    elif any(id_ == constructor_id for id_ in ("duckdb", "pyspark")):
        msg = r"\n\nHint: Did you mean one of these columns: \['a', 'b', 'z'\]?"
    else:
        msg = (
            r"The following columns were not found: \[.*\]"
            r"\n\nHint: Did you mean one of these columns: \['a', 'b', 'z'\]?"
        )

    if "polars_lazy" in str(constructor) and isinstance(df, nw.LazyFrame):
        with pytest.raises(ColumnNotFoundError, match=msg):
            df.filter(c=5).collect()
    else:
        with pytest.raises(ColumnNotFoundError, match=msg):
            df.filter(c=5)


def test_filter_with_predicates_and_constraints(
    constructor_eager: ConstructorEager,
) -> None:
    # Adapted from https://github.com/narwhals-dev/narwhals/pull/3173/commits/8433b2d75438df98004a3c850ad23628e2376836
    df = nw.from_native(constructor_eager({"a": range(5), "b": [2, 2, 4, 2, 4]}))
    mask = [True, False, True, True, False]
    mask_2 = [True, True, False, True, False]
    expected_mask_only = {"a": [0, 2, 3], "b": [2, 4, 2]}
    expected_mixed = {"a": [0, 3], "b": [2, 2]}

    result = df.filter(mask)
    assert_equal_data(result, expected_mask_only)

    msg = (
        r"unable to find column \"c\"; valid columns: \[\"a\", \"b\"\]"
        if "polars" in str(constructor_eager)
        else (
            r"The following columns were not found: \[.*\]"
            r"\n\nHint: Did you mean one of these columns: \['a', 'b'\]?"
        )
    )
    with pytest.raises(ColumnNotFoundError, match=msg):
        df.filter(mask, c=1, d=2, e=3, f=4, g=5)

    # NOTE: Everything from here is currently undefined
    result = df.filter(mask, b=2)
    assert_equal_data(result, expected_mixed)

    result = df.filter(mask, nw.col("b") == 2)
    assert_equal_data(result, expected_mixed)

    result = df.filter(mask, mask_2)
    assert_equal_data(result, expected_mixed)

    result = df.filter(
        mask, nw.Series.from_iterable("mask", mask_2, backend=df.implementation)
    )
    assert_equal_data(result, expected_mixed)

    result = df.filter(mask, nw.col("b") != 4, b=2)
    assert_equal_data(result, expected_mixed)


def test_filter_multiple_predicates(constructor: Constructor) -> None:
    """https://github.com/pola-rs/polars/blob/a4522d719de940be3ef99d494ccd1cd6067475c6/py-polars/tests/unit/lazyframe/test_lazyframe.py#L175-L202."""
    df = nw.from_native(
        constructor({"a": [1, 1, 1, 2, 2], "b": [1, 1, 2, 2, 2], "c": [1, 1, 2, 3, 4]})
    )

    # multiple predicates
    expected = {"a": [1, 1, 1], "b": [1, 1, 2], "c": [1, 1, 2]}
    for out in (
        df.filter(nw.col("a") == 1, nw.col("b") <= 2),  # positional/splat
        df.filter([nw.col("a") == 1, nw.col("b") <= 2]),  # as list
    ):
        assert_equal_data(out, expected)

    # multiple kwargs
    assert_equal_data(df.filter(a=1, b=2), {"a": [1], "b": [2], "c": [2]})

    # both positional and keyword args
    assert_equal_data(
        df.filter(nw.col("c") < 4, a=2, b=2), {"a": [2], "b": [2], "c": [3]}
    )


def test_filter_string_predicate(constructor: Constructor) -> None:
    """https://github.com/pola-rs/polars/blob/a4522d719de940be3ef99d494ccd1cd6067475c6/py-polars/tests/unit/lazyframe/test_lazyframe.py#L204-L210."""
    df = nw.from_native(
        constructor({"description": ["eq", "gt", "ge"], "predicate": ["==", ">", ">="]})
    )
    expected = {"description": ["eq"], "predicate": ["=="]}
    result = df.filter(predicate="==")
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    "predicates", [(nw.col("z") < 10,), (nw.col("a") > 0, nw.col("b") > 0)]
)
def test_filter_seq_iterable_all_true(constructor: Constructor, predicates: Any) -> None:
    """https://github.com/pola-rs/polars/blob/a4522d719de940be3ef99d494ccd1cd6067475c6/py-polars/tests/unit/lazyframe/test_lazyframe.py#L213-L233."""
    df = nw.from_native(constructor(data))
    predicate = (p for p in predicates)
    assert_equal_data(df.filter(predicate), data)


@pytest.mark.parametrize(
    "predicates", [(nw.col("z") > 10,), (nw.col("a") < 0, nw.col("b") < 0)]
)
def test_filter_seq_iterable_all_false(constructor: Constructor, predicates: Any) -> None:
    df = nw.from_native(constructor(data))
    expected: dict[str, list[Any]] = {"a": [], "b": [], "z": []}
    predicate = (p for p in predicates)
    assert_equal_data(df.filter(predicate), expected)
