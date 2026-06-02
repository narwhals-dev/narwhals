from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import pytest

import narwhals as nw
import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from narwhals.exceptions import ColumnNotFoundError, DuplicateError
from tests.plan.utils import (
    DataFrame,
    Lazy,
    LazyFrame,
    Series,
    assert_equal_data,
    assert_equal_schema,
)

if TYPE_CHECKING:
    from pytest import FixtureRequest

    from narwhals._plan.typing import IntoExpr, OneOrIterable
    from narwhals.typing import IntoDType
    from tests.conftest import Data

CheckFrameFn: TypeAlias = Callable[[DataFrame | LazyFrame], None]
"""A self-contained test that works with both `*Frame` fixtures."""

Rows: TypeAlias = list[dict[str, Any]]
"""Expected rows in a struct column."""


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "a": [1, 2, 3],
        "b": ["dogs", "cats", None],
        "c": ["play", "swim", "walk"],
        "d": [4.0, 5.0, 6.0],
    }


A, B, C, D, X, Y, Z = "a", "b", "c", "d", "x", "y", "z"
DOGS, CATS, PLAY, SWIM, WALK = "dogs", "cats", "play", "swim", "walk"
EXPECTED_1: Final = [
    {A: 1, B: DOGS, C: PLAY},
    {A: 2, B: CATS, C: SWIM},
    {A: 3, B: None, C: WALK},
]


@pytest.mark.parametrize("alias_struct", ["aliased", None])
@pytest.mark.parametrize(
    ("args", "kwds", "expected_rows"),
    [
        ((nwp.col(A), nwp.col(B), nwp.col(C)), {}, EXPECTED_1),
        (([nwp.col(A), nwp.nth(1), nwp.col(C)]), {}, EXPECTED_1),
        ((~ncs.last(),), {}, EXPECTED_1),
        ((A, B, C), {}, EXPECTED_1),
        ((), {X: A, Y: B}, [{X: 1, Y: DOGS}, {X: 2, Y: CATS}, {X: 3, Y: None}]),
        ((A,), {Z: C}, [{A: 1, Z: PLAY}, {A: 2, Z: SWIM}, {A: 3, Z: WALK}]),
        (
            (A,),
            {X: C, Y: False},
            [
                {A: 1, X: PLAY, Y: False},
                {A: 2, X: SWIM, Y: False},
                {A: 3, X: WALK, Y: False},
            ],
        ),
    ],
    ids=[
        "positional-1",
        "positional-2",
        "positional-3",
        "positional-4",
        "named",
        "positional_named",
        "literals",
    ],
)
def test_struct(
    data: Data,
    dataframe: DataFrame,
    args: tuple[OneOrIterable[IntoExpr], ...],
    kwds: dict[str, IntoExpr],
    alias_struct: str | None,
    expected_rows: Rows,
) -> None:
    expr = nwp.struct(*args, **kwds)
    if alias_struct is None:
        lhs_name = next(iter(expected_rows[0]))
        expected = {lhs_name: expected_rows}
    else:
        expr = expr.alias(alias_struct)
        expected = {alias_struct: expected_rows}
    assert_equal_data(dataframe(data).select(expr), expected)


# TODO @dangotbanned: Add a `parametrize` case that doesn't trigger things that aren't implemented too
@pytest.mark.xfail(
    reason=(
        "`NamedIR[BinaryExpr[Multiply]].resolve_dtype()` is not yet implemented, got: "
        "  [(col('a')) * (lit(2))]\n"
        "Blocked by https://github.com/narwhals-dev/narwhals/pull/3396"
    ),
    raises=NotImplementedError,
)
def test_struct_with_expressions(
    data: Data, dataframe: DataFrame, request: FixtureRequest
) -> None:
    dataframe.xfail_not_implemented(request, dataframe.is_polars(), "str.len_chars")
    df = dataframe(data)
    result = df.select(
        nwp.struct(nwp.col("a") * 2, nwp.col("c").str.len_chars()).alias("struct")
    )

    expected = {
        "struct": [{"a": 2, "c": 4}, {"a": 4, "c": 4}, {"a": 6, "c": 4}]
    }  # pragma: no cover
    assert_equal_data(result, expected)  # pragma: no cover


def test_struct_with_schema(data: Data, dataframe: DataFrame) -> None:
    dtype = nw.Struct({A: nw.Float64(), B: nw.Float32()})
    expr = nwp.struct(A, nwp.col(D).name.map(lambda _s: B)).cast(dtype)
    result = dataframe(data).select(x=expr)
    assert_equal_schema(result, {X: dtype})
    assert_equal_data(result, {X: [{A: 1.0, B: 4.0}, {A: 2.0, B: 5.0}, {A: 3.0, B: 6.0}]})


def test_struct_with_series(data: Data, series: Series) -> None:
    s_a, s_b = series(data["a"], name="a"), series(data["b"], name="b")
    result = nwp.select(struct=nwp.struct(s_a, s_b), eager=series.implementation)

    expected = {
        "struct": [{"a": 1, "b": "dogs"}, {"a": 2, "b": "cats"}, {"a": 3, "b": None}]
    }

    assert_equal_data(result, expected)


def test_struct_mixed_series_and_exprs(data: Data, dataframe: DataFrame) -> None:
    df = dataframe(data)
    s_a = df.get_column("a")
    result = df.select(nwp.struct(s_a, nwp.col("c")).alias("struct"))

    expected = {
        "struct": [{"a": 1, "c": "play"}, {"a": 2, "c": "swim"}, {"a": 3, "c": "walk"}]
    }

    assert_equal_data(result, expected)


def test_struct_named_with_series(data: Data, dataframe: DataFrame) -> None:
    df = dataframe(data)
    s_a = df.get_column("a")
    result = df.select(nwp.struct(x=s_a, y="b").alias("struct"))

    expected = {
        "struct": [{"x": 1, "y": "dogs"}, {"x": 2, "y": "cats"}, {"x": 3, "y": None}]
    }

    assert_equal_data(result, expected)


def test_error_on_duplicate_field_name_22959(lazy: Lazy) -> None:
    # https://github.com/pola-rs/polars/blob/346a793589efd552a6c10c857e0f0434f7e9a7d4/py-polars/tests/unit/functions/as_datatype/test_struct.py#L270-L277
    with pytest.raises(DuplicateError, match="'literal'"):
        nwp.select(nwp.struct(nwp.lit(1), nwp.lit(2)), lazy=lazy).collect_schema()


@pytest.mark.parametrize(
    ("exprs", "expected_fields"),
    [
        (
            (
                ncs.last(),
                ncs.boolean(),
                ncs.string(),
                ncs.float(),
                ncs.integer() - ncs.by_dtype(nw.UInt32),
            ),
            {
                "e": nw.UInt32,
                "c": nw.Boolean,
                "b": nw.String,
                "d": nw.Float64,
                "a": nw.Int64,
            },
        )
    ],
)
@pytest.mark.parametrize("alias_struct", [None, "struct", "d"])
def test_struct_select_lazy_schema(
    lazy: Lazy,
    exprs: OneOrIterable[IntoExpr],
    expected_fields: dict[str, IntoDType],
    alias_struct: str | None,
) -> None:
    lf = nwp.select(
        a=1, b=nwp.lit("2"), c=False, d=1.3, e=nwp.lit(8).cast(nw.UInt32), lazy=lazy
    )

    if alias_struct:
        struct = nwp.struct(exprs).alias(alias_struct)
        name_outer = alias_struct
    else:
        struct = nwp.struct(exprs)
        name_outer = next(iter(expected_fields))

    expected = {name_outer: nw.Struct(expected_fields)}
    assert_equal_schema(lf.select(struct), expected)


@pytest.mark.xfail(
    reason=("TODO @dangotbanned: Add nested `struct()` cases"), raises=NotImplementedError
)
def test_struct_nested() -> None:
    raise NotImplementedError


def test_struct_broadcasting(dataframe: DataFrame) -> None:
    # https://github.com/pola-rs/polars/blob/346a793589efd552a6c10c857e0f0434f7e9a7d4/py-polars/tests/unit/functions/as_datatype/test_struct.py#L174-L193
    data = {"col1": [1, 2], "col2": [10, 20]}
    df = dataframe(data)
    expr = nwp.struct([nwp.lit("a").alias("a"), nwp.col("col1").alias("col1")]).alias(
        "my_struct"
    )
    result = df.select(expr)
    expected = {"my_struct": [{"a": "a", "col1": 1}, {"a": "a", "col1": 2}]}
    assert_equal_data(result, expected)


def test_suffix_in_struct_creation(dataframe: DataFrame) -> None:
    # https://github.com/pola-rs/polars/blob/346a793589efd552a6c10c857e0f0434f7e9a7d4/py-polars/tests/unit/functions/as_datatype/test_struct.py#L240-L249
    data = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
    df = dataframe(data)
    expr = nwp.struct(nwp.col(["a", "c"]).name.suffix("_foo")).alias("bar")
    result = df.select(expr).unnest("bar")
    expected = {"a_foo": [1, 2], "c_foo": [5, 6]}
    assert_equal_data(result, expected)


@pytest.fixture
def query_15442() -> CheckFrameFn:
    # https://github.com/pola-rs/polars/blob/346a793589efd552a6c10c857e0f0434f7e9a7d4/py-polars/tests/unit/functions/as_datatype/test_struct.py#L252-L267
    center = nwp.struct(x=nwp.col("x"), y=nwp.col("y"))
    left = 0
    in_x = (left < center.struct.field("x")) & (center.struct.field("x") <= 1000)

    def check(fixture: DataFrame | LazyFrame, /) -> None:
        result = fixture({"x": [206.0], "y": [225.0]}).filter(in_x)
        if isinstance(result, nwp.LazyFrame):
            result = result.collect()
        assert result.shape == (1, 2)

    return check


def test_resolved_names_15442(dataframe: DataFrame, query_15442: CheckFrameFn) -> None:
    query_15442(dataframe)


def test_resolved_names_15442_lazy(
    lazyframe: LazyFrame, query_15442: CheckFrameFn, request: FixtureRequest
) -> None:
    # NOTE: Upstream test is only on LazyFrame
    lazyframe.xfail_not_implemented(request, lazyframe.is_polars(), "LazyFrame.filter")
    query_15442(lazyframe)


def test_raise_subnodes_18787(dataframe: DataFrame) -> None:
    # https://github.com/pola-rs/polars/blob/346a793589efd552a6c10c857e0f0434f7e9a7d4/py-polars/tests/unit/lazyframe/test_schema.py#L416-L424
    df = dataframe({"a": [1], "b": [2]})
    bad = ncs.first().struct.field("a").filter(nwp.col("foo") == 1)
    with pytest.raises(ColumnNotFoundError, match=r"'foo'"):
        df.select(nwp.struct(nwp.all())).select(bad)
