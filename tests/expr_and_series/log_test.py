from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 2, 4]}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_log_expr(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").log(base=2))
    expected = {"a": [0, 1, 2]}
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_log_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.log(base=2)
    expected = {"a": [0, 1, 2]}
    assert_equal_data({"a": result}, expected)
