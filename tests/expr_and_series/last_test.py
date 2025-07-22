from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from narwhals.typing import PythonLiteral
    from tests.utils import ConstructorEager

data: dict[str, list[PythonLiteral]] = {
    "a": [8, 2, 1, None],
    "b": [58, 5, 6, 12],
    "c": [2.5, 1.0, 3.0, 0.9],
    "d": [2, 1, 4, 3],
    "idx": [0, 1, 2, 3],
}

single_cases = pytest.mark.parametrize(
    ("col", "expected"), [("a", None), ("b", 12), ("c", 0.9)]
)


@single_cases
def test_last_series(
    constructor_eager: ConstructorEager, col: str, expected: PythonLiteral
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    result = series.last()
    assert_equal_data({col: [result]}, {col: [expected]})


def test_last_series_empty(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["b"]
    series = series.filter(series > 60)
    result = series.last()
    assert result is None


@single_cases
def test_last_expr_select(
    constructor_eager: ConstructorEager, col: str, expected: PythonLiteral
) -> None:
    df = nw.from_native(constructor_eager(data))
    expr = nw.col(col).last()
    result = df.select(expr)
    assert_equal_data(result, {col: [expected]})


@single_cases
def test_last_expr_with_columns(
    constructor_eager: ConstructorEager,
    col: str,
    expected: PythonLiteral,
    request: pytest.FixtureRequest,
) -> None:
    if any(
        x in str(constructor_eager) for x in ("pyarrow_table", "pandas", "modin", "cudf")
    ):
        request.applymarker(
            pytest.mark.xfail(
                reason="Some kind of index error, see https://github.com/narwhals-dev/narwhals/pull/2528#discussion_r2083582828"
            )
        )

    request.applymarker(
        pytest.mark.xfail(
            ("polars" in str(constructor_eager) and POLARS_VERSION < (1, 10)),
            reason="Needs `order_by`",
            raises=NotImplementedError,
        )
    )

    frame = nw.from_native(constructor_eager(data))
    expr = nw.col(col).last().over(order_by="idx").alias("result")
    result = frame.with_columns(expr).select("result")
    expected_broadcast = len(data[col]) * [expected]
    assert_equal_data(result, {"result": expected_broadcast})


@pytest.mark.parametrize(
    "expected",
    [
        {"a": [None], "c": [0.9]},
        {"d": [3], "b": [12]},
        {"c": [0.9], "a": [None], "d": [3]},
    ],
)
def test_last_expr_expand(
    constructor_eager: ConstructorEager, expected: Mapping[str, Sequence[PythonLiteral]]
) -> None:
    df = nw.from_native(constructor_eager(data))
    expr = nw.col(expected).last()
    result = df.select(expr)
    assert_equal_data(result, expected)


def test_last_expr_expand_sort(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    expr = nw.col("d", "a", "b", "c").last()
    result = df.sort("d").select(expr)
    expected = {"d": [4], "a": [1], "b": [6], "c": [3.0]}
    assert_equal_data(result, expected)
