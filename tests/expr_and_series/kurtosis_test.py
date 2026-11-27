from __future__ import annotations

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import (
    DASK_PYARROW_DTYPES,
    PANDAS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    xfail_if_builtin_dask,
)


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

    df = nw.from_native(constructor({"a": data}))
    if data and PANDAS_VERSION < PkgVersion("3"):
        xfail_if_builtin_dask(df, request, when=DASK_PYARROW_DTYPES)
    result = df.select(nw.col("a").kurtosis())
    assert_equal_data(result, {"a": [expected]})


def test_arrow_kurtosis_ignores_nulls(
    arrow_constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    df = nw.from_native(arrow_constructor({"a": [1, None, 2]}))
    xfail_if_builtin_dask(df, request)
    assert_equal_data(df.select(nw.col("a").kurtosis()), {"a": [-2.0]})
