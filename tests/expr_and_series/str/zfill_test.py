from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    uses_pyarrow_backend,
)

data = {"a": ["-1", "+1", "1", "12", "123", "99999", "+9999", None]}
expected = {"a": ["-01", "+01", "001", "012", "123", "99999", "+9999", None]}


def test_str_zfill(
    request: pytest.FixtureRequest, nw_frame_constructor: Constructor
) -> None:
    if uses_pyarrow_backend(nw_frame_constructor) and PANDAS_VERSION < (3,):
        reason = (
            "pandas with pyarrow backend doesn't support str.zfill, see "
            "https://github.com/pandas-dev/pandas/issues/61485"
        )
        request.applymarker(pytest.mark.xfail(reason=reason))

    if "pandas" in str(nw_frame_constructor) and PANDAS_VERSION < (1, 5):
        reason = "different zfill behavior"
        pytest.skip(reason=reason)

    if "polars" in str(nw_frame_constructor) and POLARS_VERSION < (0, 20, 5):
        reason = (
            "`TypeError: argument 'length': 'Expr' object cannot be interpreted as an integer`"
            "in `expr.str.slice(1, length)`"
        )
        pytest.skip(reason=reason)

    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(nw.col("a").str.zfill(3))
    assert_equal_data(result, expected)


def test_str_zfill_series(
    request: pytest.FixtureRequest, nw_eager_constructor: ConstructorEager
) -> None:
    if uses_pyarrow_backend(nw_eager_constructor) and PANDAS_VERSION < (3,):
        reason = (
            "pandas with pyarrow backend doesn't support str.zfill, see "
            "https://github.com/pandas-dev/pandas/issues/61485"
        )
        request.applymarker(pytest.mark.xfail(reason=reason))

    if "pandas" in str(nw_eager_constructor) and PANDAS_VERSION < (1, 5):
        reason = "different zfill behavior"
        pytest.skip(reason=reason)

    if "polars" in str(nw_eager_constructor) and POLARS_VERSION < (0, 20, 5):
        reason = (
            "`TypeError: argument 'length': 'Expr' object cannot be interpreted as an integer`"
            "in `expr.str.slice(1, length)`"
        )
        pytest.skip(reason=reason)

    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    result = df["a"].str.zfill(3)
    assert_equal_data({"a": result}, expected)
