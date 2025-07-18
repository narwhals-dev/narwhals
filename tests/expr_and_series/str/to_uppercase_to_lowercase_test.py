from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


@pytest.mark.parametrize(
    ("data", "expected"), [({"a": ["foo", "bar"]}, {"a": ["FOO", "BAR"]})]
)
def test_str_to_uppercase(
    constructor: Constructor, data: dict[str, list[str]], expected: dict[str, list[str]]
) -> None:
    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("a").str.to_uppercase())

    assert_equal_data(result_frame, expected)


@pytest.mark.parametrize(
    ("data", "expected"), [({"a": ["foo", "bar"]}, {"a": ["FOO", "BAR"]})]
)
def test_str_to_uppercase_series(
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    expected: dict[str, list[str]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
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
