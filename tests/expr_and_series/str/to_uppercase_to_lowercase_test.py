from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"a": ["foo", "bar"]}, {"a": ["FOO", "BAR"]}),
        (
            {
                "a": [
                    "special case ß",
                    "ςpecial caσe",  # noqa: RUF001
                ]
            },
            {"a": ["SPECIAL CASE ẞ", "ΣPECIAL CAΣE"]},
        ),
    ],
)
def test_str_to_uppercase(
    constructor: Constructor, data: dict[str, list[str]], expected: dict[str, list[str]]
) -> None:
    if (
        any(
            x in str(constructor)
            for x in (
                "pandas_constructor",
                "pandas_nullable",
                "polars",
                "cudf",
                "pyspark",
            )
        )
        and "ẞ" in expected["a"][0]
        and "sqlframe" not in str(constructor)
    ):
        expected = {"a": ["SPECIAL CASE SS", "ΣPECIAL CAΣE"]}

    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("a").str.to_uppercase())

    assert_equal_data(result_frame, expected)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"a": ["foo", "bar"]}, {"a": ["FOO", "BAR"]}),
        (
            {
                "a": [
                    "special case ß",
                    "ςpecial caσe",  # noqa: RUF001
                ]
            },
            {"a": ["SPECIAL CASE ẞ", "ΣPECIAL CAΣE"]},
        ),
    ],
)
def test_str_to_uppercase_series(
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    if (
        any(
            x in str(constructor_eager)
            for x in (
                "pandas_constructor",
                "pandas_nullable",
                "polars",
                "cudf",
                "pyspark",
            )
        )
        and "ẞ" in expected["a"][0]
        and "sqlframe" not in str(constructor_eager)
    ):
        expected = {"a": ["SPECIAL CASE SS", "ΣPECIAL CAΣE"]}

    result_series = df["a"].str.to_uppercase()
    assert_equal_data({"a": result_series}, expected)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"a": ["FOO", "BAR"]}, {"a": ["foo", "bar"]}),
        (
            {"a": ["SPECIAL CASE ß", "ΣPECIAL CAΣE"]},
            {
                "a": [
                    "special case ß",
                    "σpecial caσe",  # noqa: RUF001
                ]
            },
        ),
    ],
)
def test_str_to_lowercase(
    constructor: Constructor, data: dict[str, list[str]], expected: dict[str, list[str]]
) -> None:
    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("a").str.to_lowercase())
    assert_equal_data(result_frame, expected)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"a": ["FOO", "BAR"]}, {"a": ["foo", "bar"]}),
        (
            {"a": ["SPECIAL CASE ß", "ΣPECIAL CAΣE"]},
            {
                "a": [
                    "special case ß",
                    "σpecial caσe",  # noqa: RUF001
                ]
            },
        ),
    ],
)
def test_str_to_lowercase_series(
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.to_lowercase()
    assert_equal_data({"a": result_series}, expected)
