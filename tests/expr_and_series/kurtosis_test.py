from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], None),
        ([1], None),
        ([1, 2], -2.0),
        ([0.0, 0.0, 0.0], None),
        ([1, 2, 3, 2, 1], -1.153061),
    ],
)
def test_kurtosis_series(
    constructor_eager: ConstructorEager, data: list[float], expected: float | None
) -> None:
    result = nw.from_native(constructor_eager({"a": data}), eager_only=True)[
        "a"
    ].kurtosis()
    assert_equal_data({"a": [result]}, {"a": [expected]})


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], None),
        ([1], None),
        ([1, 2], -2.0),
        ([0.0, 0.0, 0.0], None),
        ([1, 2, 3, 2, 1], -1.153061),
    ],
    ids=range(5),
)
@pytest.mark.filterwarnings("ignore:.*invalid value:RuntimeWarning:dask")
def test_kurtosis_expr(
    constructor: Constructor,
    data: list[float],
    expected: float | None,
    request: pytest.FixtureRequest,
) -> None:
    if "ibis" in str(constructor):
        # https://github.com/ibis-project/ibis/issues/11341
        request.applymarker(pytest.mark.xfail)

    if "pyspark" in str(constructor) and int(request.node.callspec.id[-1]) == 0:
        # Can not infer schema from empty dataset.
        pytest.skip()

    result = nw.from_native(constructor({"a": data})).select(nw.col("a").kurtosis())
    assert_equal_data(result, {"a": [expected]})
