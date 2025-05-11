from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Mapping
from typing import Sequence

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION
from tests.utils import POLARS_VERSION
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import PythonLiteral
    from tests.utils import Constructor
    from tests.utils import ConstructorEager

data: dict[str, list[PythonLiteral]] = {
    "a": [8, 2, 1, None],
    "b": [58, 5, 6, 12],
    "c": [2.5, 1.0, 3.0, 0.9],
    "d": [2, 1, 4, 3],
    "idx": [0, 1, 2, 3],
}


@pytest.mark.parametrize(("col", "expected"), [("a", 8), ("b", 58), ("c", 2.5)])
def test_first_series(
    constructor_eager: ConstructorEager, col: str, expected: PythonLiteral
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    result = series.first()
    assert_equal_data({col: [result]}, {col: [expected]})


def test_first_series_empty(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    series = series.filter(series > 50)
    result = series.first()
    assert result is None


@pytest.mark.parametrize(("col", "expected"), [("a", 8), ("b", 58), ("c", 2.5)])
def test_first_expr_eager(
    constructor_eager: ConstructorEager, col: str, expected: PythonLiteral
) -> None:
    df = nw.from_native(constructor_eager(data))
    expr = nw.col(col).first()
    result = df.select(expr)
    assert_equal_data(result, {col: [expected]})


@pytest.mark.skip(
    reason="https://github.com/narwhals-dev/narwhals/pull/2528#discussion_r2083557149"
)
@pytest.mark.parametrize(("col", "expected"), [("a", 8), ("b", 58), ("c", 2.5)])
def test_first_expr_lazy_select(
    constructor: Constructor, col: str, expected: PythonLiteral
) -> None:  # pragma: no cover
    frame = nw.from_native(constructor(data))
    expr = nw.col(col).first().over(order_by="idx")
    result = frame.select(expr)
    assert_equal_data(result, {col: [expected]})


@pytest.mark.parametrize(("col", "expected"), [("a", 8), ("b", 58), ("c", 2.5)])
def test_first_expr_lazy_with_columns(
    constructor: Constructor,
    col: str,
    expected: PythonLiteral,
    request: pytest.FixtureRequest,
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "pandas", "modin", "cudf")):
        request.applymarker(
            pytest.mark.xfail(
                reason="Some kind of index error, see https://github.com/narwhals-dev/narwhals/pull/2528#discussion_r2083582828"
            )
        )

    request.applymarker(
        pytest.mark.xfail(
            ("duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3)),
            reason="Needs `SQLExpression`",
            raises=NotImplementedError,
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            ("polars" in str(constructor) and POLARS_VERSION < (1, 10)),
            reason="Needs `order_by`",
            raises=NotImplementedError,
        )
    )

    frame = nw.from_native(constructor(data))
    expr = nw.col(col).first().over(order_by="idx").alias("result")
    result = frame.with_columns(expr).select("result")
    expected_broadcast = len(data[col]) * [expected]
    assert_equal_data(result, {"result": expected_broadcast})


@pytest.mark.parametrize(
    "expected",
    [{"a": [8], "c": [2.5]}, {"d": [2], "b": [58]}, {"c": [2.5], "a": [8], "d": [2]}],
)
def test_first_expr_eager_expand(
    constructor_eager: ConstructorEager, expected: Mapping[str, Sequence[PythonLiteral]]
) -> None:
    df = nw.from_native(constructor_eager(data))
    expr = nw.col(expected).first()
    result = df.select(expr)
    assert_equal_data(result, expected)


def test_first_expr_eager_expand_sort(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    expr = nw.col("d", "a", "b", "c").first()
    result = df.sort("d").select(expr)
    expected = {"d": [1], "a": [2], "b": [5], "c": [1.0]}
    assert_equal_data(result, expected)
