from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import pytest

import narwhals as nw
import narwhals._plan.demo as nwd
from narwhals._plan.expr import Column, _ColumnSelection
from narwhals.exceptions import ColumnNotFoundError

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._plan.common import ExprIR
    from narwhals._plan.dummy import DummyExpr
    from narwhals.dtypes import DType


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
