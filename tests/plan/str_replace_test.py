from __future__ import annotations

from typing import Final

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

A1: Final = ["123abc", "abc456"]
A2: Final = ["abc abc", "abc456"]
A3: Final = ["abc abc abc", "456abc"]
A4: Final = ["Dollar $ign", "literal"]
B: Final = ["ghi", "jkl"]
XFAIL_STR_REPLACE_EXPR = pytest.mark.xfail(
    reason="`replace(value:Expr, n>1)` is not yet supported for `pyarrow`",
    raises=NotImplementedError,
)
XFAIL_STR_REPLACE_ALL_EXPR = pytest.mark.xfail(
    reason="`replace_all(value:Expr)` is not yet supported for `pyarrow`",
    raises=NotImplementedError,
)

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
        (A1, r"abc", nwp.col("b"), 1, False, ["123ghi", "jkl456"]),
        (A2, r"abc", nwp.col("b"), 1, False, ["ghi abc", "jkl456"]),
        pytest.param(
            A3,
            r"abc",
            nwp.col("b"),
            -1,
            False,
            ["ghi ghi ghi", "456jkl"],
            marks=XFAIL_STR_REPLACE_EXPR,
        ),
        pytest.param(
            A4,
            r"$",
            nwp.col("b"),
            -1,
            True,
            ["Dollar ghiign", "literal"],
            marks=XFAIL_STR_REPLACE_EXPR,
        ),
        (
            ["dogcatdogcat", "dog dog"],
            "cat",
            nwp.col("b").last(),
            1,
            True,
            ["dogjkldogcat", "dog dog"],
        ),
        (
            A3,
            r"^abc",
            nwp.col("b").str.to_uppercase(),
            1,
            False,
            ["GHI abc abc", "456abc"],
        ),
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
        (A1, r"abc", "b", False, ["123ghi", "jkl456"]),
        (A2, r"abc", "b", False, ["ghi ghi", "jkl456"]),
        (A4, r"$", "b", True, ["Dollar ghiign", "literal"]),
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
    data: list[str],
    pattern: str,
    value: nwp.Expr,
    n: int,
    *,
    literal: bool,
    expected: list[str],
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


@XFAIL_STR_REPLACE_ALL_EXPR
@replace_all_vector
def test_str_replace_all_vector(
    data: list[str], pattern: str, value: str, *, literal: bool, expected: list[str]
) -> None:  # pragma: no cover
    df = dataframe({"a": data, "b": B})
    result = df.select(
        nwp.col("a").str.replace_all(pattern, nwp.col(value), literal=literal)
    )
    assert_equal_data(result, {"a": expected})
