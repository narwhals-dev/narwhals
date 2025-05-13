from __future__ import annotations

# ruff: noqa: FBT001
from typing import TYPE_CHECKING
from typing import Sequence

import pytest

import narwhals as nw
from narwhals.exceptions import ColumnNotFoundError
from tests.utils import POLARS_VERSION
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import PythonLiteral
    from tests.utils import ConstructorEager

_EXPECTED_A = 0, 0, 2, -1, 1

data = {
    "a": _EXPECTED_A,
    "b": [1, 3, 2, None, None],
    "c": [None, None, 12.0, 5.5, 12.1],
    "d": [4, 0, 1, 3, 2],
    "group_1": ["C", "D", "D", "E", "E"],
    "group_2": [None, "G", "F", "G", None],
}


@pytest.mark.parametrize(
    ("cols", "by", "descending", "nulls_last", "expected"),
    [
        ("group_1", "d", False, False, {"group_1": ["D", "D", "E", "E", "C"]}),
        ("group_1", ["c", "b"], False, True, {"group_1": ["E", "D", "E", "C", "D"]}),
        (
            ["d", "a"],
            ["group_2", "b"],
            [True, False],
            False,
            {"d": [2, 4, 3, 0, 1], "a": [1, 0, -1, 0, 2]},
        ),
        (
            ["d", "a"],
            ["group_2", "b"],
            [True, False],
            True,
            {"d": [0, 3, 1, 4, 2], "a": [0, -1, 2, 0, 1]},
        ),
        (
            ["c", "a", "group_2"],
            ["group_1", "b", "d"],
            [True, False, False],
            False,
            {
                "c": [12.1, 5.5, 12.0, None, None],
                "a": [1, -1, 2, 0, 0],
                "group_2": [None, "G", "F", "G", None],
            },
        ),
    ],
)
def test_sort_by(
    constructor_eager: ConstructorEager,
    cols: str | Sequence[str],
    by: str | Sequence[str],
    descending: bool | Sequence[bool],
    nulls_last: bool,
    expected: dict[str, Sequence[PythonLiteral]],
    request: pytest.FixtureRequest,
) -> None:
    request.applymarker(
        pytest.mark.xfail(
            "polars" in str(constructor_eager)
            and POLARS_VERSION < (0, 20, 31)
            and nulls_last,
            reason="Missing `nulls_last` support",
            raises=NotImplementedError,
        )
    )
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        nw.col(cols).sort_by(by, descending=descending, nulls_last=nulls_last)
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, {"a": _EXPECTED_A, "b": [3, 2, 1, None, None]}),
        (True, False, {"a": _EXPECTED_A, "b": [None, None, 3, 2, 1]}),
        (False, True, {"a": _EXPECTED_A, "b": [1, 2, 3, None, None]}),
        (False, False, {"a": _EXPECTED_A, "b": [None, None, 1, 2, 3]}),
    ],
)
def test_sort_by_self(
    constructor_eager: ConstructorEager,
    descending: bool | Sequence[bool],
    nulls_last: bool,
    expected: dict[str, Sequence[PythonLiteral]],
    request: pytest.FixtureRequest,
) -> None:
    request.applymarker(
        pytest.mark.xfail(
            "polars" in str(constructor_eager)
            and POLARS_VERSION < (0, 20, 31)
            and nulls_last,
            reason="Missing `nulls_last` support",
            raises=NotImplementedError,
        )
    )
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        "a", nw.col("b").sort_by("b", descending=descending, nulls_last=nulls_last)
    )
    assert_equal_data(result, expected)


def test_sort_by_invalid(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises((ColumnNotFoundError, KeyError), match="NON EXIST"):
        df.select(nw.col("b", "group_2").sort_by("group_1", "NON EXIST"))
    with pytest.raises((ColumnNotFoundError, KeyError), match="NON EXIST"):
        df.with_columns(
            nw.col("b", "group_2").sort_by("group_1", "NON EXIST").name.suffix("_suffix")
        )


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nw.col("a").sort_by("i").arg_min(), {"a": [1]}),
        (nw.col("a").sort_by("i").arg_max(), {"a": [0]}),
    ],
)
def test_sort_by_orderable_agg(
    constructor_eager: ConstructorEager,
    expr: nw.Expr,
    expected: dict[str, Sequence[PythonLiteral]],
) -> None:
    data = {"a": [9, 8, 7], "i": [0, 2, 1]}
    df = nw.from_native(constructor_eager(data)).sort("a", descending=False)
    result = df.select(expr)
    assert_equal_data(result, expected)
