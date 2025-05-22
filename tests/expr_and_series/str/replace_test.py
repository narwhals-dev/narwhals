from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

replace_data = [
    ({"a": ["123abc", "abc456"]}, r"abc\b", "ABC", 1, False, {"a": ["123ABC", "abc456"]}),
    ({"a": ["abc abc", "abc456"]}, r"abc", "", 1, False, {"a": [" abc", "456"]}),
    ({"a": ["abc abc abc", "456abc"]}, r"abc", "", -1, False, {"a": ["  ", "456"]}),
    (
        {"a": ["Dollar $ign", "literal"]},
        r"$",
        "S",
        -1,
        True,
        {"a": ["Dollar Sign", "literal"]},
    ),
]

replace_all_data = [
    ({"a": ["123abc", "abc456"]}, r"abc\b", "ABC", False, {"a": ["123ABC", "abc456"]}),
    ({"a": ["abc abc", "abc456"]}, r"abc", "", False, {"a": [" ", "456"]}),
    ({"a": ["abc abc abc", "456abc"]}, r"abc", "", False, {"a": ["  ", "456"]}),
    (
        {"a": ["Dollar $ign", "literal"]},
        r"$",
        "S",
        True,
        {"a": ["Dollar Sign", "literal"]},
    ),
]


@pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"), replace_data
)
def test_str_replace_series(
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    n: int,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.replace(
        pattern=pattern, value=value, n=n, literal=literal
    )
    assert_equal_data({"a": result_series}, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "literal", "expected"), replace_all_data
)
def test_str_replace_all_series(
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.replace_all(pattern=pattern, value=value, literal=literal)
    assert_equal_data({"a": result_series}, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"), replace_data
)
def test_str_replace_expr(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    n: int,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
) -> None:
    if (
        ("pyspark" in str(constructor))
        or "duckdb" in str(constructor)
        or "ibis" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result_df = df.select(
        nw.col("a").str.replace(pattern=pattern, value=value, n=n, literal=literal)
    )
    assert_equal_data(result_df, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "literal", "expected"), replace_all_data
)
def test_str_replace_all_expr(
    constructor: Constructor,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("a").str.replace_all(pattern=pattern, value=value, literal=literal)
    )
    assert_equal_data(result, expected)
