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

replace_data_multivalue = [
    (
        {"a": ["123abc", "abc456"], "b": ["ghi", "jkl"]},
        r"abc",
        "b",
        1,
        False,
        {"a": ["123ghi", "jkl456"]},
    ),
    (
        {"a": ["abc abc", "abc456"], "b": ["ghi", "jkl"]},
        r"abc",
        "b",
        1,
        False,
        {"a": ["ghi abc", "jkl456"]},
    ),
    (
        {"a": ["abc abc abc", "456abc"], "b": ["ghi", "jkl"]},
        r"abc",
        "b",
        -1,
        False,
        {"a": ["ghi ghi ghi", "456jkl"]},
    ),
    (
        {"a": ["Dollar $ign", "literal"], "b": ["ghi", "jkl"]},
        r"$",
        "b",
        -1,
        True,
        {"a": ["Dollar ghiign", "literal"]},
    ),
]

replace_all_data_multivalue = [
    (
        {"a": ["123abc", "abc456"], "b": ["ghi", "jkl"]},
        r"abc",
        "b",
        False,
        {"a": ["123ghi", "jkl456"]},
    ),
    (
        {"a": ["abc abc", "abc456"], "b": ["ghi", "jkl"]},
        r"abc",
        "b",
        False,
        {"a": ["ghi ghi", "jkl456"]},
    ),
    (
        {"a": ["Dollar $ign", "literal"], "b": ["ghi", "jkl"]},
        r"$",
        "b",
        True,
        {"a": ["Dollar ghiign", "literal"]},
    ),
]


@pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"), replace_data
)
def test_str_replace_series_scalar(
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
def test_str_replace_all_series_scalar(
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
def test_str_replace_expr_scalar(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    n: int,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
) -> None:
    if any(x in str(constructor) for x in ["pyspark", "duckdb", "ibis"]):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor} only supports `replace_all`.",
                raises=NotImplementedError,
            )
        )
    df = nw.from_native(constructor(data))
    result_df = df.select(
        nw.col("a").str.replace(pattern=pattern, value=value, n=n, literal=literal)
    )
    assert_equal_data(result_df, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "literal", "expected"), replace_all_data
)
def test_str_replace_all_expr_scalar(
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


@pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"), replace_data_multivalue
)
def test_str_replace_series_multivalue(
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    n: int,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
    request: pytest.FixtureRequest,
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor_eager} does not support multivalue replacement",
                raises=TypeError,
            )
        )

    result_series = df["a"].str.replace(
        pattern=pattern, value=df[value], n=n, literal=literal
    )
    assert_equal_data({"a": result_series}, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "literal", "expected"), replace_all_data_multivalue
)
def test_str_replace_all_series_multivalue(
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
    request: pytest.FixtureRequest,
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor_eager} only supports `replace_all`.",
                raises=TypeError,
            )
        )

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result_series = df["a"].str.replace_all(
        pattern=pattern, value=df[value], literal=literal
    )
    assert_equal_data({"a": result_series}, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "n", "literal", "expected"), replace_data_multivalue
)
def test_str_replace_expr_multivalue(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    n: int,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
) -> None:
    if any(x in str(constructor) for x in ["pyspark", "duckdb", "ibis"]):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor} only supports `replace_all`.",
                raises=NotImplementedError,
            )
        )
    elif any(x in str(constructor) for x in ["pyarrow_table", "dask"]):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor} does not support multivalue replacement",
                raises=TypeError,
            )
        )

    df = nw.from_native(constructor(data))
    result_df = df.select(
        nw.col("a").str.replace(
            pattern=pattern, value=nw.col(value), n=n, literal=literal
        )
    )
    assert_equal_data(result_df, expected)


@pytest.mark.parametrize(
    ("data", "pattern", "value", "literal", "expected"), replace_all_data_multivalue
)
def test_str_replace_all_expr_multivalue(
    constructor: Constructor,
    data: dict[str, list[str]],
    pattern: str,
    value: str,
    literal: bool,  # noqa: FBT001
    expected: dict[str, list[str]],
    request: pytest.FixtureRequest,
) -> None:
    if any(x in str(constructor) for x in ["pyarrow_table", "pyspark", "ibis", "dask"]):
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{constructor} does not support multivalue replacement",
                raises=TypeError,
            )
        )

    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("a").str.replace_all(pattern=pattern, value=nw.col(value), literal=literal)
    )
    assert_equal_data(result, expected)
