from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [None, 1, 2, None, 4, 6, 11]}
expected = {
    "x1": [float("nan")] * 6 + [21],
    "x2": [float("nan"), 1.0, 3.0, 3.0, 6.0, 10.0, 21.0],
    "x3": [float("nan"), 1.0, 3.0, 2.0, 4.0, 10.0, 17.0],
    "x4": [3.0, 3.0, 7.0, 13.0, 23.0, 21.0, 21.0],
}


@pytest.mark.filterwarnings(
    "ignore:`Expr.rolling_sum` is being called from the stable API although considered an unstable feature."
)
def test_rolling_sum_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "dask" in str(constructor):
        # TODO(FBruzzesi): Dask is raising the following error:
        # NotImplementedError: Partition size is less than overlapping window size.
        # Try using ``df.repartition`` to increase the partition size.
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(
        x1=nw.col("a").rolling_sum(window_size=3),
        x2=nw.col("a").rolling_sum(window_size=3, min_periods=1),
        x3=nw.col("a").rolling_sum(window_size=2, min_periods=1),
        x4=nw.col("a").rolling_sum(window_size=5, min_periods=1, center=True),
    )

    assert_equal_data(result, expected)


@pytest.mark.filterwarnings(
    "ignore:`Series.rolling_sum` is being called from the stable API although considered an unstable feature."
)
def test_rolling_sum_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result = df.select(
        x1=df["a"].rolling_sum(window_size=3),
        x2=df["a"].rolling_sum(window_size=3, min_periods=1),
        x3=df["a"].rolling_sum(window_size=2, min_periods=1),
        x4=df["a"].rolling_sum(window_size=5, min_periods=1, center=True),
    )
    assert_equal_data(result, expected)
