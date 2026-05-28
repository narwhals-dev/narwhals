from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import pytest

import narwhals._plan as nwp
from tests.plan.utils import DataFrame, Series, assert_equal_data, assert_equal_series

if TYPE_CHECKING:
    from tests.conftest import Data


class Kwds(TypedDict, total=False):
    com: float | None
    span: float | None
    half_life: float | None
    alpha: float | None
    adjust: bool
    min_samples: int
    ignore_nulls: bool


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [1, 1, 2], "b": [1, 2, 3]}


@pytest.mark.parametrize(
    ("kwds", "expected"),
    [
        (Kwds(com=1), {"a": [1.0, 1.0, 1.571429], "b": [1.0, 1.666667, 2.428571]}),
        (Kwds(com=1, adjust=False), {"a": [1.0, 1.0, 1.5], "b": [1.0, 1.5, 2.25]}),
    ],
)
def test_ewm_mean_expr(
    data: Data,
    kwds: Kwds,
    expected: dict[str, list[float]],
    dataframe: DataFrame,
    request: pytest.FixtureRequest,
) -> None:
    dataframe.xfail_not_implemented(request, dataframe.is_pyarrow(), "Expr.ewm_mean")
    result = dataframe(data).select(nwp.col("a", "b").ewm_mean(**kwds))
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("kwds", "expected"),
    [
        (Kwds(com=1, ignore_nulls=True), {"a": [2.0, 3.333333, None, 3.142857]}),
        (Kwds(com=1), {"a": [2.0, 3.333333, None, 3.090909]}),
    ],
)
def test_ewm_mean_nulls(
    request: pytest.FixtureRequest,
    kwds: Kwds,
    expected: dict[str, list[float]],
    dataframe: DataFrame,
) -> None:
    dataframe.xfail_not_implemented(request, dataframe.is_pyarrow(), "Expr.ewm_mean")
    df = dataframe({"a": [2.0, 4.0, None, 3.0]})
    result = df.select(nwp.col("a").ewm_mean(**kwds))
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("kwds", "expected"),
    [
        (Kwds(alpha=0.5, ignore_nulls=True), {"a": [2.0, 4.0, 3.428571]}),
        (Kwds(span=1.5, ignore_nulls=True), {"a": [2.0, 4.5, 3.290323]}),
        (Kwds(half_life=1.5, adjust=False), {"a": [2.0, 3.110118, 3.069370]}),
        (Kwds(alpha=0.5, min_samples=2, ignore_nulls=True), {"a": [None, 4.0, 3.428571]}),
    ],
)
def test_ewm_mean_params(
    kwds: Kwds, expected: Data, request: pytest.FixtureRequest, dataframe: DataFrame
) -> None:
    dataframe.xfail_not_implemented(request, dataframe.is_pyarrow(), "Expr.ewm_mean")
    df = dataframe({"a": [2, 5, 3]})
    assert_equal_data(df.select(nwp.col("a").ewm_mean(**kwds)), expected)


@pytest.mark.xfail(reason="TODO @dangotbanned: `Series.ewm_mean`", raises=AttributeError)
def test_ewm_mean_series(
    data: Data, request: pytest.FixtureRequest, series: Series
) -> None:  # pragma: no cover
    series.xfail_not_implemented(request, series.is_pyarrow(), "Series.ewm_mean")
    result = series(data["a"], name="a").ewm_mean(com=1)  # type: ignore[attr-defined]
    expected = [1.0, 1.0, 1.571429]
    assert_equal_series(result, expected, "a")
