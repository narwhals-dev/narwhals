from __future__ import annotations

from math import sqrt
from typing import Any

import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {"a": [1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0]}

kwargs_and_expected = (
    {
        "name": "x1",
        "kwargs": {"window_size": 3},
        "expected": [
            sqrt(x) if x is not None else x
            for x in [None, None, 1 / 3, 1, 4 / 3, 7 / 3, 3]
        ],
    },
    {
        "name": "x2",
        "kwargs": {"window_size": 3, "min_samples": 1},
        "expected": [
            sqrt(x) if x is not None else x
            for x in [None, 0.5, 1 / 3, 1.0, 4 / 3, 7 / 3, 3]
        ],
    },
    {
        "name": "x3",
        "kwargs": {"window_size": 2, "min_samples": 1},
        "expected": [
            sqrt(x) if x is not None else x for x in [None, 0.5, 0.5, 2.0, 2.0, 4.5, 4.5]
        ],
    },
    {
        "name": "x4",
        "kwargs": {"window_size": 5, "min_samples": 1, "center": True},
        "expected": [
            sqrt(x) if x is not None else x
            for x in [1 / 3, 11 / 12, 4 / 5, 17 / 10, 2.0, 2.25, 3]
        ],
    },
    {
        "name": "x5",
        "kwargs": {"window_size": 4, "min_samples": 1, "center": True},
        "expected": [
            sqrt(x) if x is not None else x
            for x in [0.5, 1 / 3, 11 / 12, 11 / 12, 2.25, 2.25, 3]
        ],
    },
    {
        "name": "x6",
        "kwargs": {"window_size": 3, "ddof": 2},
        "expected": [
            sqrt(x) if x is not None else x
            for x in [None, None, 2 / 3, 2.0, 8 / 3, 14 / 3, 6.0]
        ],
    },
)


@pytest.mark.parametrize("kwargs_and_expected", kwargs_and_expected)
def test_rolling_std_expr(
    constructor_eager: ConstructorEager, kwargs_and_expected: dict[str, Any]
) -> None:
    name = kwargs_and_expected["name"]
    kwargs = kwargs_and_expected["kwargs"]
    expected = kwargs_and_expected["expected"]

    if "polars" in str(constructor_eager) and POLARS_VERSION < (1,):
        pytest.skip()

    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").rolling_std(**kwargs).alias(name))

    assert_equal_data(result, {name: expected})


@pytest.mark.filterwarnings(
    "ignore:`Series.rolling_std` is being called from the stable API although considered an unstable feature."
)
@pytest.mark.parametrize("kwargs_and_expected", kwargs_and_expected)
def test_rolling_std_series(
    constructor_eager: ConstructorEager, kwargs_and_expected: dict[str, Any]
) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1,):
        pytest.skip()

    name = kwargs_and_expected["name"]
    kwargs = kwargs_and_expected["kwargs"]
    expected = kwargs_and_expected["expected"]

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].rolling_std(**kwargs).alias(name))

    assert_equal_data(result, {name: expected})


@pytest.mark.parametrize(
    ("expected_a", "window_size", "min_samples", "center", "ddof"),
    [
        ([None, None, 0.5, None, None, 1, 2.5], 2, None, False, 0),
        (
            [
                None,
                None,
                0.7071067811865476,
                None,
                None,
                1.4142135623730951,
                3.5355339059327378,
            ],
            2,
            2,
            False,
            1,
        ),
        (
            [
                None,
                None,
                0.7071067811865476,
                0.7071067811865476,
                1.4142135623730951,
                1.4142135623730951,
                3.605551275463989,
            ],
            3,
            2,
            False,
            1,
        ),
        ([0.0, None, 0.5, 0.5, 1.0, 1.0, 2.943920288775949], 3, 1, False, 0),
        (
            [
                0.7071067811865476,
                None,
                0.7071067811865476,
                1.4142135623730951,
                1.4142135623730951,
                3.605551275463989,
                3.5355339059327378,
            ],
            3,
            1,
            True,
            1,
        ),
        (
            [
                0.7071067811865476,
                None,
                0.7071067811865476,
                1.5275252316519465,
                2.0,
                3.605551275463989,
                3.605551275463989,
            ],
            4,
            1,
            True,
            1,
        ),
        (
            [
                0.5,
                0.5,
                1.247219128924647,
                1.920286436967152,
                3.344772040064913,
                2.943920288775949,
                2.943920288775949,
            ],
            5,
            1,
            True,
            0,
        ),
    ],
)
def test_rolling_std_expr_lazy_ungrouped(
    constructor: Constructor,
    expected_a: list[float],
    window_size: int,
    min_samples: int,
    *,
    center: bool,
    ddof: int,
) -> None:
    if ("polars" in str(constructor) and POLARS_VERSION < (1, 10)) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3)
    ):
        pytest.skip()
    if "modin" in str(constructor):
        # unreliable
        pytest.skip()
    if "dask" in str(constructor) and ddof != 1:
        # Only `ddof=1` is supported
        pytest.skip()
    data = {
        "a": [1, None, 2, None, 4, 6, 11],
        "b": [1, None, 2, 3, 4, 5, 6],
        "i": list(range(7)),
    }
    df = nw.from_native(constructor(data))
    result = (
        df.with_columns(
            nw.col("a")
            .rolling_std(window_size, min_samples=min_samples, center=center, ddof=ddof)
            .over(order_by="b")
        )
        .select("a", "i")
        .sort("i")
    )
    expected = {"a": expected_a, "i": list(range(7))}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("expected_a", "window_size", "min_samples", "center", "ddof"),
    [
        ([None, None, 0.5, None, None, 1.0, 2.5], 2, None, False, 0),
        (
            [
                None,
                None,
                0.7071067811865476,
                None,
                None,
                1.4142135623730951,
                3.5355339059327378,
            ],
            2,
            2,
            False,
            1,
        ),
        (
            [
                None,
                None,
                0.7071067811865476,
                0.7071067811865476,
                None,
                1.4142135623730951,
                3.605551275463989,
            ],
            3,
            2,
            False,
            1,
        ),
        ([0.0, None, 0.5, 0.5, 0.0, 1.0, 2.943920288775949], 3, 1, False, 0),
        (
            [
                0.7071067811865476,
                None,
                0.7071067811865476,
                None,
                1.4142135623730951,
                3.605551275463989,
                3.5355339059327378,
            ],
            3,
            1,
            True,
            1,
        ),
        (
            [
                0.7071067811865476,
                None,
                0.7071067811865476,
                0.7071067811865476,
                1.4142135623730951,
                3.605551275463989,
                3.605551275463989,
            ],
            4,
            1,
            True,
            1,
        ),
        (
            [0.5, 0.5, 0.5, 0.5, 2.943920288775949, 2.943920288775949, 2.943920288775949],
            5,
            1,
            True,
            0,
        ),
    ],
)
def test_rolling_std_expr_lazy_grouped(
    constructor: Constructor,
    expected_a: list[float],
    window_size: int,
    min_samples: int,
    request: pytest.FixtureRequest,
    *,
    center: bool,
    ddof: int,
) -> None:
    if (
        ("polars" in str(constructor) and POLARS_VERSION < (1, 10))
        or ("duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3))
        or ("pandas" in str(constructor) and PANDAS_VERSION < (1, 2))
    ):
        pytest.skip()
    if any(x in str(constructor) for x in ("dask", "pyarrow_table")):
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(constructor):
        # unreliable
        pytest.skip()
    data = {
        "a": [1, None, 2, None, 4, 6, 11],
        "g": [1, 1, 1, 1, 2, 2, 2],
        "b": [1, None, 2, 3, 4, 5, 6],
        "i": list(range(7)),
    }
    df = nw.from_native(constructor(data))
    result = (
        df.with_columns(
            nw.col("a")
            .rolling_std(window_size, min_samples=min_samples, center=center, ddof=ddof)
            .over("g", order_by="b")
        )
        .sort("i")
        .select("a")
    )
    expected = {"a": expected_a}
    assert_equal_data(result, expected)
