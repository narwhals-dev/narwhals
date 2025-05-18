from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


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
    if "ibis" in str(constructor):
        # https://github.com/ibis-project/ibis/issues/11176
        request.applymarker(pytest.mark.xfail)

    if "pyspark" in str(constructor) and int(request.node.callspec.id[-1]) == 0:
        # Can not infer schema from empty dataset.
        pytest.skip()

    result = nw.from_native(constructor({"a": data})).select(nw.col("a").skew())
    assert_equal_data(result, {"a": [expected]})
