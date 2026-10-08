from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION, Constructor, ConstructorEager, assert_equal_data


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], None),
        ([1], None),
        ([1, 2], 0.0),
        ([0.0, 0.0, 0.0], None),
        ([1, 2, 3, 2, 1], 0.343622),
    ],
)
def test_skew_series(
    constructor_eager: ConstructorEager, data: list[float], expected: float | None
) -> None:
    result = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].skew()
    assert_equal_data({"a": [result]}, {"a": [expected]})


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], None),
        ([1], None),
        ([1, 2], 0.0),
        ([0.0, 0.0, 0.0], None),
        ([1, 2, 3, 2, 1], 0.343622),
    ],
    ids=range(5),
)
@pytest.mark.filterwarnings("ignore:.*invalid value:RuntimeWarning:dask")
def test_skew_expr(
    constructor: Constructor,
    data: list[float],
    expected: float | None,
    request: pytest.FixtureRequest,
) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()

    if "ibis" in str(constructor) and int(request.node.callspec.id[-1]) == 0:
        # Can not infer schema from empty dataset.
        pytest.skip()

    if "pyspark" in str(constructor) and int(request.node.callspec.id[-1]) == 0:
        # Can not infer schema from empty dataset.
        pytest.skip()

    result = nw.from_native(constructor({"a": data})).select(nw.col("a").skew())
    assert_equal_data(result, {"a": [expected]})
    result = nw.from_native(constructor({"a": data})).with_columns(nw.col("a").skew())
    assert_equal_data(result, {"a": [expected] * len(data)})


def test_skew_large_offset(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    unstable_backends = ("duckdb", "sqlframe")
    if any(backend in str(constructor) for backend in unstable_backends):
        request.applymarker(
            pytest.mark.xfail(reason="Native skew is numerically unstable")
        )

    offset = 1e9
    df = nw.from_native(
        constructor(
            {"value": [offset + 1, offset + 2, offset + 3, offset + 2, offset + 1]}
        )
    )
    result = df.select(nw.col("value").skew())
    assert_equal_data(result, {"value": [0.343622]})
