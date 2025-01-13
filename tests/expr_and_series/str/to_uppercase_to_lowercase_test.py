from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import PYARROW_VERSION
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


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
            {"a": ["SPECIAL CASE SS", "ΣPECIAL CAΣE"]},
        ),
    ],
)
def test_str_to_uppercase(
    constructor: ConstructorEager,
    data: dict[str, list[str]],
    expected: dict[str, list[str]],
    request: pytest.FixtureRequest,
) -> None:
    if any("ß" in s for value in data.values() for s in value) & (
        constructor.__name__
        in (
            "pandas_pyarrow_constructor",
            "pyarrow_table_constructor",
            "modin_pyarrow_constructor",
            "duckdb_lazy_constructor",
        )
        or ("dask" in str(constructor) and PYARROW_VERSION >= (12,))
    ):
        # We are marking it xfail for these conditions above
        # since the pyarrow backend will convert
        # smaller cap 'ß' to upper cap 'ẞ' instead of 'SS'
        request.applymarker(pytest.mark.xfail)

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
            {"a": ["SPECIAL CASE SS", "ΣPECIAL CAΣE"]},
        ),
    ],
)
def test_str_to_uppercase_series(
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    expected: dict[str, list[str]],
    request: pytest.FixtureRequest,
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    if any("ß" in s for value in data.values() for s in value) & (
        constructor_eager.__name__
        not in (
            "pandas_constructor",
            "pandas_nullable_constructor",
            "polars_eager_constructor",
            "cudf_constructor",
            "duckdb_lazy_constructor",
            "modin_constructor",
        )
    ):
        # We are marking it xfail for these conditions above
        # since the pyarrow backend will convert
        # smaller cap 'ß' to upper cap 'ẞ' instead of 'SS'
        request.applymarker(pytest.mark.xfail)

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
    constructor: ConstructorEager,
    data: dict[str, list[str]],
    expected: dict[str, list[str]],
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
