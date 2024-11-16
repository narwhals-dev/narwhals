from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION
from tests.utils import PYARROW_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [3, 1, None, 2]}

expected = {
    "cum_min": [3, 1, None, 1],
    "reverse_cum_min": [1, 1, None, 2],
}


@pytest.mark.parametrize("reverse", [True, False])
def test_cum_min_expr(
    request: pytest.FixtureRequest, constructor: Constructor, *, reverse: bool
) -> None:
    if "dask" in str(constructor) and reverse:
        request.applymarker(pytest.mark.xfail)

    if (PANDAS_VERSION < (2, 1) or PYARROW_VERSION < (13,)) and "pandas_pyarrow" in str(
        constructor
    ):
        request.applymarker(pytest.mark.xfail)

    if (PANDAS_VERSION < (2, 1) or PYARROW_VERSION < (13,)) and "pandas_pyarrow" in str(
        constructor
    ):
        request.applymarker(pytest.mark.xfail)

    name = "reverse_cum_min" if reverse else "cum_min"
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("a").cum_min(reverse=reverse).alias(name),
    )

    assert_equal_data(result, {name: expected[name]})


def test_cum_min_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if (PANDAS_VERSION < (2, 1) or PYARROW_VERSION < (13,)) and "pandas_pyarrow" in str(
        constructor_eager
    ):
        request.applymarker(pytest.mark.xfail)

    if (PANDAS_VERSION < (2, 1) or PYARROW_VERSION < (13,)) and "pandas_pyarrow" in str(
        constructor_eager
    ):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        cum_min=df["a"].cum_min(),
        reverse_cum_min=df["a"].cum_min(reverse=True),
    )
    assert_equal_data(result, expected)
