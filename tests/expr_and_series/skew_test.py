from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


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
)
@pytest.mark.filterwarnings("ignore:.*invalid value:RuntimeWarning:dask")
def test_skew_expr(
    constructor: Constructor,
    data: list[float],
    expected: float | None,
    request: pytest.FixtureRequest,
) -> None:
    if any(x in str(constructor) for x in ("ibis",)):
        # https://github.com/ibis-project/ibis/issues/11176
        request.applymarker(pytest.mark.xfail)

    if any(x in str(constructor) for x in ("pyspark", "ibis", "daft")) and not data:
        # Can not infer schema from empty dataset.
        pytest.skip()

    result = nw.from_native(constructor({"a": data})).select(nw.col("a").skew())
    assert_equal_data(result, {"a": [expected]})
