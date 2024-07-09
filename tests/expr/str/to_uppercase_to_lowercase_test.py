from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


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
    constructor: Any, data: Any | None, expected: Any, request: Any
) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result_frame = df.select(nw.col("a").str.to_uppercase())
    if (constructor.__name__ == "pandas_pyarrow_constructor") & df["a"].str.contains(
        "ß"
    ).any():
        request.applymarker(pytest.mark.xfail)
    compare_dicts(result_frame, expected)

    result_series = df["a"].str.to_uppercase()
    assert result_series.to_numpy().tolist() == expected["a"]


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
def test_str_to_lowercase(constructor: Any, data: Any | None, expected: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result_frame = df.select(nw.col("a").str.to_lowercase())
    compare_dicts(result_frame, expected)

    result_series = df["a"].str.to_lowercase()
    assert result_series.to_numpy().tolist() == expected["a"]
