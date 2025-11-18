from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [1, 1, 1, 2, 2, 3], "b": [1, 2, 3, 4, 5, 6]}


def test_any_value_expr(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        reason = "sample does not allow n, use frac instead"
        request.applymarker(pytest.mark.xfail(reason=reason))

    df = nw.from_native(constructor(data))

    # Aggregation
    result = df.select(nw.col("a", "b").any_value()).lazy().collect()
    assert result.shape == (1, 2)

    # Aggregation + broadcast
    result = df.select(nw.col("a"), nw.col("b").any_value()).lazy().collect()
    assert result.shape == (6, 2)
    assert result["b"].n_unique() == 1

    # Aggregation + broadcast
    result = df.select(nw.col("a").any_value(), nw.col("b")).lazy().collect()
    assert result.shape == (6, 2)
    assert result["a"].n_unique() == 1


def test_any_value_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result = {"a": [df["a"].any_value()], "b": [df["b"].any_value()]}
    assert all(len(r) == 1 for r in result.values())


def test_any_value_group_by(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        reason = "sample does not allow n, use frac instead"
        request.applymarker(pytest.mark.xfail(reason=reason))

    df = nw.from_native(constructor(data))

    result = df.group_by("a").agg(nw.col("b").any_value().alias("any"), nw.col("b").max())
    assert result.lazy().collect().shape == (3, 3)


def test_any_value_over(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        reason = "sample does not allow n, use frac instead"
        request.applymarker(pytest.mark.xfail(reason=reason))

    df = nw.from_native(constructor(data))

    result = df.select(nw.col("a"), nw.col("b").any_value().over("a"))
    assert result.lazy().collect().shape == (6, 2)
    uniques = result.group_by("a").agg(nw.col("b").n_unique()).sort("a")
    assert_equal_data(uniques, {"a": [1, 2, 3], "b": [1, 1, 1]})
