from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import pytest
from tests.utils import (
    PANDAS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import DTypeBackend

data = {"a": [-pi, -pi / 2, 0.0, pi / 2, pi]}

expected = [-1, 0, 1, 0, -1]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_cos_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").cos())
    assert_equal_data(result, {"a": expected})


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_cos_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.cos()
    assert_equal_data({"a": result}, {"a": expected})


PYARROW_UNAVAILABLE = PYARROW_VERSION == (0, 0, 0)
reason = "nullable types require pandas2+"
require_pd_2_1 = pytest.mark.skipif(
    PANDAS_VERSION < (2, 1, 0) or PYARROW_UNAVAILABLE, reason=reason
)
require_pd_2_0 = pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason=reason)


@pytest.mark.parametrize(
    "dtype_backend",
    [
        pytest.param("pyarrow", marks=require_pd_2_1),
        pytest.param("numpy_nullable", marks=require_pd_2_0),
        None,
    ],
)
def test_cos_dtype_pandas(dtype_backend: DTypeBackend) -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s = pd.Series([-pi / 2, None, pi / 2], name="a", dtype="float32", index=[8, 7, 6])
    if dtype_backend:
        s = s.convert_dtypes(dtype_backend=dtype_backend)
    result = nw.from_native(s, series_only=True).cos().to_native()
    expected = pd.Series([0, None, 0], name="a", dtype=s.dtype, index=[8, 7, 6])
    pd.testing.assert_series_equal(result, expected, rtol=0, atol=1e-6)


@pytest.mark.skipif(PANDAS_VERSION < (2, 1, 0), reason="nullable types require pandas2+")
def test_cos_dtype_pandas_pyarrow() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    s = pd.Series(
        [-pi / 2, None, pi / 2], name="a", dtype="Float32[pyarrow]", index=[8, 7, 6]
    )
    result = nw.from_native(s, series_only=True).cos().to_native()
    expected = pd.Series(
        [0, None, 0], name="a", dtype="Float32[pyarrow]", index=[8, 7, 6]
    )
    pd.testing.assert_series_equal(result, expected, rtol=0, atol=1e-6)
