from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import pytest

import narwhals as nw
import narwhals._plan.demo as nwd
from narwhals._plan.expr import Alias, Column, _ColumnSelection
from narwhals._plan.expr_expansion import rewrite_special_aliases
from narwhals.exceptions import ColumnNotFoundError, ComputeError

if TYPE_CHECKING:
    from typing_extensions import TypeAlias, TypeIs

    from narwhals._plan.common import ExprIR
    from narwhals._plan.dummy import DummyExpr
    from narwhals.dtypes import DType


MapIR: TypeAlias = "Callable[[ExprIR], ExprIR]"


@pytest.fixture
def schema_1() -> dict[str, DType]:
    return {
        "a": nw.Int64(),
        "b": nw.Int32(),
        "c": nw.Int16(),
        "d": nw.Int8(),
        "e": nw.UInt64(),
        "f": nw.UInt32(),
        "g": nw.UInt16(),
        "h": nw.UInt8(),
        "i": nw.Float64(),
        "j": nw.Float32(),
        "k": nw.String(),
        "l": nw.Datetime(),
        "m": nw.Boolean(),
        "n": nw.Date(),
        "o": nw.Datetime(),
        "p": nw.Categorical(),
        "q": nw.Duration(),
        "r": nw.Enum(["A", "B", "C"]),
        "s": nw.List(nw.String()),
        "u": nw.Struct({"a": nw.Int64(), "k": nw.String()}),
    }


# NOTE: The meta check doesn't provide typing and describes a superset of `_ColumnSelection`
def is_column_selection(obj: ExprIR) -> TypeIs[_ColumnSelection]:
    return obj.meta.is_column_selection(allow_aliasing=False) and isinstance(
        obj, _ColumnSelection
    )


def seq_column_from_names(names: Sequence[str]) -> tuple[Column, ...]:
    return tuple(Column(name=name) for name in names)


@pytest.mark.parametrize(
    ("expr", "into_expected"),
    [
        (nwd.col("a", "c"), ["a", "c"]),
        (nwd.col("o", "k", "b"), ["o", "k", "b"]),
        (nwd.nth(5), ["f"]),
        (nwd.nth(0, 1, 2, 3, 4), ["a", "b", "c", "d", "e"]),
        (nwd.nth(-1), ["u"]),
        (nwd.nth([-2, -3, -4]), ["s", "r", "q"]),
        (
            nwd.all(),
            [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "u",
            ],
        ),
        (
            nwd.exclude("a", "c", "e", "l", "q"),
            ["b", "d", "f", "g", "h", "i", "j", "k", "m", "n", "o", "p", "r", "s", "u"],
        ),
    ],
)
def test_expand_columns_root(
    expr: DummyExpr, into_expected: Sequence[str], schema_1: dict[str, DType]
) -> None:
    expected = seq_column_from_names(into_expected)
    selection = expr._ir
    assert is_column_selection(selection)
    actual = selection.expand_columns(schema_1)
    assert actual == expected


@pytest.mark.parametrize(
    "expr",
    [
        nwd.col("y", "z"),
        nwd.col("a", "b", "z"),
        nwd.col("x", "b", "a"),
        nwd.col(
            [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "FIVE",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "u",
            ]
        ),
    ],
)
def test_invalid_expand_columns(expr: DummyExpr, schema_1: dict[str, DType]) -> None:
    selection = expr._ir
    assert is_column_selection(selection)
    with pytest.raises(ColumnNotFoundError):
        selection.expand_columns(schema_1)


def udf_name_map(name: str) -> str:
    original = name
    upper = name.upper()
    lower = name.lower()
    title = name.title()
    return f"{original=} | {upper=} | {lower=} | {title=}"


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwd.col("a").name.to_uppercase(), "A"),
        (nwd.col("B").name.to_lowercase(), "b"),
        (nwd.col("c").name.suffix("_after"), "c_after"),
        (nwd.col("d").name.prefix("before_"), "before_d"),
        (
            nwd.col("aBcD EFg hi").name.map(udf_name_map),
            "original='aBcD EFg hi' | upper='ABCD EFG HI' | lower='abcd efg hi' | title='Abcd Efg Hi'",
        ),
        (nwd.col("a").min().alias("b").over("c").alias("d").max().name.keep(), "a"),
        (
            (
                nwd.col("hello")
                .sort_by(nwd.col("ignore me"))
                .max()
                .over("ignore me as well")
                .first()
                .name.to_uppercase()
            ),
            "HELLO",
        ),
        (
            (
                nwd.col("start")
                .alias("next")
                .sort()
                .round()
                .fill_null(5)
                .alias("noise")
                .name.suffix("_end")
            ),
            "start_end",
        ),
    ],
)
def test_rewrite_special_aliases_single(expr: DummyExpr, expected: str) -> None:
    # NOTE: We can't use `output_name()` without resolving these rewrites
    # Once they're done, `output_name()` just peeks into `Alias(name=...)`
    ir_input = expr._ir
    with pytest.raises(ComputeError):
        ir_input.meta.output_name()

    ir_output = rewrite_special_aliases(ir_input)
    assert ir_input != ir_output
    actual = ir_output.meta.output_name()
    assert actual == expected


def alias_replace_guarded(name: str) -> MapIR:  # pragma: no cover
    """Guards against repeatedly creating the same alias."""

    def fn(ir: ExprIR) -> ExprIR:
        if isinstance(ir, Alias) and ir.name != name:
            return Alias(expr=ir.expr, name=name)
        return ir

    return fn


def alias_replace_unguarded(name: str) -> MapIR:  # pragma: no cover
    """**Does not guard against recursion**!

    Handling the recursion stopping **should be** part of the impl of `ExprIR.map_ir`.

    - *Ideally*, return an identical result to `alias_replace_guarded` (after the same number of iterations)
    - *Pragmatically*, it might require an extra iteration to detect a cycle
    """

    def fn(ir: ExprIR) -> ExprIR:
        if isinstance(ir, Alias):
            return Alias(expr=ir.expr, name=name)
        return ir

    return fn


@pytest.mark.xfail(
    reason="Not implemented `ExprIR.map_ir` yet", raises=NotImplementedError
)
@pytest.mark.parametrize(
    ("expr", "function", "into_expected"),
    [
        (nwd.col("a"), alias_replace_guarded("never"), nwd.col("a")),
        (nwd.col("a"), alias_replace_unguarded("never"), nwd.col("a")),
        (nwd.col("a").alias("b"), alias_replace_guarded("c"), nwd.col("a").alias("c")),
        (nwd.col("a").alias("b"), alias_replace_unguarded("c"), nwd.col("a").alias("c")),
        (
            nwd.col("a").alias("d").first().over("b", order_by="c").alias("e"),
            alias_replace_guarded("d"),
            nwd.col("a").alias("d").first().over("b", order_by="c").alias("d"),
        ),
        (
            nwd.col("a").alias("d").first().over("b", order_by="c").alias("e"),
            alias_replace_unguarded("d"),
            nwd.col("a").alias("d").first().over("b", order_by="c").alias("d"),
        ),
    ],
)
def test_map_ir_recursive(
    expr: DummyExpr, function: MapIR, into_expected: DummyExpr
) -> None:
    ir = expr._ir
    expected = into_expected._ir
    actual = ir.map_ir(function)
    assert actual == expected  # pragma: no cover
