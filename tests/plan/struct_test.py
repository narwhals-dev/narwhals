from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from narwhals.exceptions import DuplicateError
from tests.plan.utils import DataFrame, Lazy, assert_equal_data, assert_equal_schema

if TYPE_CHECKING:
    from pytest import FixtureRequest

    from narwhals._plan.typing import IntoExpr, OneOrIterable
    from narwhals.typing import IntoDType
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [1, 2, 3], "b": ["dogs", "cats", None], "c": ["play", "swim", "walk"]}


@pytest.mark.parametrize(
    "exprs",
    [
        (nwp.col("a"), nwp.col("b"), nwp.col("c")),
        ([nwp.col("a"), nwp.col("b"), nwp.col("c")]),
        (nwp.all(),),
        ("a", "b", "c"),
    ],
)
def test_struct_positional_exprs(
    data: Data, dataframe: DataFrame, exprs: tuple[nwp.Expr | list[nwp.Expr], ...]
) -> None:
    df = dataframe(data)
    result = df.select(nwp.struct(*exprs))
    expected = {
        "a": [
            {"a": 1, "b": "dogs", "c": "play"},
            {"a": 2, "b": "cats", "c": "swim"},
            {"a": 3, "b": None, "c": "walk"},
        ]
    }

    assert_equal_data(result, expected)


@pytest.mark.parametrize("alias_struct", ["struct", None])
def test_struct_named_exprs(
    data: Data, dataframe: DataFrame, alias_struct: str | None
) -> None:
    df = dataframe(data)
    expr = nwp.struct(x="a", y="b")
    rows = [{"x": 1, "y": "dogs"}, {"x": 2, "y": "cats"}, {"x": 3, "y": None}]
    if alias_struct is None:
        # NOTE: This isn't a distinct case on `main`, but more of a minimal repro for expansion issue
        # The outer aliasing already worked, but the inner bits are
        expected = {"x": rows}
    else:
        expected = {alias_struct: rows}
        expr = expr.alias(alias_struct)
    assert_equal_data(df.select(expr), expected)


def test_struct_positional_and_named(data: Data, dataframe: DataFrame) -> None:
    df = dataframe(data)
    result = df.select(nwp.struct("a", z="c").alias("struct"))

    expected = {
        "struct": [{"a": 1, "z": "play"}, {"a": 2, "z": "swim"}, {"a": 3, "z": "walk"}]
    }

    assert_equal_data(result, expected)


def test_struct_with_expressions(
    data: Data, dataframe: DataFrame, request: FixtureRequest
) -> None:
    dataframe.xfail_not_implemented(request, dataframe.is_polars(), "str.len_chars")
    df = dataframe(data)
    result = df.select(
        nwp.struct(nwp.col("a") * 2, nwp.col("c").str.len_chars()).alias("struct")
    )

    expected = {"struct": [{"a": 2, "c": 4}, {"a": 4, "c": 4}, {"a": 6, "c": 4}]}
    assert_equal_data(result, expected)


def test_struct_with_literals(data: Data, dataframe: DataFrame) -> None:

    df = dataframe(data)
    result = df.select(nwp.struct("a", x="c", y=nwp.lit(False)).alias("struct"))

    expected = {
        "struct": [
            {"a": 1, "x": "play", "y": False},
            {"a": 2, "x": "swim", "y": False},
            {"a": 3, "x": "walk", "y": False},
        ]
    }

    assert_equal_data(result, expected)


def test_struct_with_schema(dataframe: DataFrame) -> None:
    data_numeric = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    schema = {"a": nw.Float64(), "b": nw.Float32()}
    df = dataframe(data_numeric)
    result = df.select(nwp.struct("a", "b").cast(nw.Struct(schema)).alias("struct"))
    assert result.collect_schema()["struct"] == nw.Struct(schema)

    expected = {
        "struct": [{"a": 1.0, "b": 4.0}, {"a": 2.0, "b": 5.0}, {"a": 3.0, "b": 6.0}]
    }
    assert_equal_data(result, expected)


def test_struct_with_series(data: Data, dataframe: DataFrame) -> None:
    df = dataframe(data)
    s_a, s_b = df.get_column("a"), df.get_column("b")
    result = df.select(nwp.struct(s_a, s_b).alias("struct"))

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
