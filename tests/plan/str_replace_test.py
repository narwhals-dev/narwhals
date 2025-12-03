from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Sequence

A1: Final = ["123abc", "abc456"]
A2: Final = ["abc abc", "abc456"]
A3: Final = ["abc abc abc", "456abc"]
A4: Final = ["Dollar $ign", "literal"]
A5: Final = [None, "oop"]
B: Final = ["ghi", "jkl"]


replace_scalar = pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"),
    [
        (A1, r"abc\b", "ABC", 1, False, ["123ABC", "abc456"]),
        (A2, r"abc", "", 1, False, [" abc", "456"]),
        (A3, r"abc", "", -1, False, ["  ", "456"]),
        (A4, r"$", "S", -1, True, ["Dollar Sign", "literal"]),
    ],
)
replace_vector = pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"),
    [
        pytest.param(
            A1, r"abc", nwp.col("b"), 1, False, ["123ghi", "jkl456"], id="n-1-single"
        ),
        pytest.param(
            A2, r"abc", nwp.col("b"), 1, False, ["ghi abc", "jkl456"], id="n-1-mixed"
        ),
        pytest.param(
            A3,
            r"a",
            nwp.col("b"),
            2,
            False,
            ["ghibc ghibc abc", "456jklbc"],
            id="n-2-mixed",
            marks=pytest.mark.xfail(
                reason="TODO: str.replace(value: Expr, n>1)", raises=NotImplementedError
            ),
        ),
        pytest.param(
            A3,
            r"abc",
            nwp.col("b"),
            -1,
            False,
            ["ghi ghi ghi", "456jkl"],
            id="replace_all",
        ),
        pytest.param(
            A4,
            r"$",
            nwp.col("b"),
            -1,
            True,
            ["Dollar ghiign", "literal"],
            id="literal-replace_all",
        ),
        pytest.param(
            ["dogcatdogcat", "dog dog"],
            "cat",
            nwp.col("b").last(),
            1,
            True,
            ["dogjkldogcat", "dog dog"],
            id="agg-replacement",
        ),
        pytest.param(
            A3,
            r"^abc",
            nwp.col("b").str.to_uppercase(),
            1,
            False,
            ["GHI abc abc", "456abc"],
            id="transformed-replacement",
        ),
        pytest.param(A5, r"o", nwp.col("b"), 1, False, [None, "jklop"], id="null-input"),
    ],
)
replace_all_scalar = pytest.mark.parametrize(
    ("data", "pattern", "value", "literal", "expected"),
    [
        (A1, r"abc\b", "ABC", False, ["123ABC", "abc456"]),
        (A2, r"abc", "", False, [" ", "456"]),
        (A3, r"abc", "", False, ["  ", "456"]),
        (A4, r"$", "S", True, ["Dollar Sign", "literal"]),
    ],
)


replace_all_vector = pytest.mark.parametrize(
    ("data", "pattern", "value", "literal", "expected"),
    [
        pytest.param(A1, r"abc", nwp.col("b"), False, ["123ghi", "jkl456"], id="single"),
        pytest.param(A2, r"abc", nwp.col("b"), False, ["ghi ghi", "jkl456"], id="mixed"),
        pytest.param(
            A4, r"$", nwp.col("b"), True, ["Dollar ghiign", "literal"], id="literal"
        ),
        pytest.param(A5, r"o", nwp.col("b"), False, [None, "jkljklp"], id="null-input"),
        pytest.param(
            A3,
            r"\d",
            nwp.col("b").first(),
            False,
            ["abc abc abc", "ghighighiabc"],
            id="agg-replacement",
        ),
        pytest.param(
            A3,
            r" ?abc$",
            nwp.lit(" HELLO").str.to_lowercase().str.strip_chars(),
            False,
            ["abc abchello", "456hello"],
            id="transformed-replacement",
        ),
    ],
)


@replace_scalar
def test_str_replace_scalar(
    data: list[str],
    pattern: str,
    value: str,
    n: int,
    *,
    literal: bool,
    expected: list[str],
) -> None:
    df = dataframe({"a": data})
    result = df.select(nwp.col("a").str.replace(pattern, value, n=n, literal=literal))
    assert_equal_data(result, {"a": expected})


@replace_vector
def test_str_replace_vector(
    data: Sequence[str | None],
    pattern: str,
    value: nwp.Expr,
    n: int,
    *,
    literal: bool,
    expected: Sequence[str | None],
) -> None:
    df = dataframe({"a": data, "b": B})
    result = df.select(nwp.col("a").str.replace(pattern, value, n=n, literal=literal))
    assert_equal_data(result, {"a": expected})


@replace_all_scalar
def test_str_replace_all_scalar(
    data: list[str], pattern: str, value: str, *, literal: bool, expected: list[str]
) -> None:
    df = dataframe({"a": data})
    result = df.select(nwp.col("a").str.replace_all(pattern, value, literal=literal))
    assert_equal_data(result, {"a": expected})


@replace_all_vector
def test_str_replace_all_vector(
    data: Sequence[str | None],
    pattern: str,
    value: nwp.Expr,
    *,
    literal: bool,
    expected: Sequence[str | None],
) -> None:
    df = dataframe({"a": data, "b": B})
    result = df.select(nwp.col("a").str.replace_all(pattern, value, literal=literal))
    assert_equal_data(result, {"a": expected})
