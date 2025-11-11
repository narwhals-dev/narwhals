from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.conftest import MIN_PANDAS_NULLABLE_VERSION
from tests.utils import PANDAS_VERSION, Constructor, assert_equal_data


@pytest.mark.parametrize("n", [0, 1, 10])
def test_clear(request: pytest.FixtureRequest, constructor: Constructor, n: int) -> None:
    if n > 0 and any(
        x in str(constructor)
        for x in (
            "cudf",
            "dask",
            "pandas_constructor",
            # Column "str" has type "string[python]", mapped to None.
            # See `test_clear_pandas_nullable` for a test that has all nullable dtypes.
            "pandas_nullable",
            "modin_constructor",
        )
    ):
        reason = "NotImplementedError"
        request.applymarker(pytest.mark.xfail(reason))

    data = {
        "int": [1, 2, 3],
        "str": ["foo", "bar", "baz"],
        "float": [0.1, 0.2, 0.3],
        "bool": [True, False, True],
    }
    df = nw.from_native(constructor(data))

    df_clear = df.clear(n=n).lazy().collect()
    assert len(df_clear) == n
    assert df.collect_schema() == df_clear.collect_schema()

    assert_equal_data(df_clear, {k: [None] * n for k in data})


def test_clear_negative(constructor: Constructor) -> None:
    n = -1
    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor(data))

    msg = f"`n` should be greater than or equal to 0, got {n}"
    with pytest.raises(ValueError, match=msg):
        df.clear(n=n)


@pytest.mark.parametrize("n", ["foo", 2.0, 1 + 1j])
def test_clear_non_integer(constructor: Constructor, n: Any) -> None:
    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor(data))

    msg = "`n` should be an integer, got type"
    with pytest.raises(TypeError, match=msg):
        df.clear(n=n)


@pytest.mark.skipif(
    PANDAS_VERSION < MIN_PANDAS_NULLABLE_VERSION, reason="too old for nullable"
)
@pytest.mark.parametrize("n", [0, 1, 10])
def test_clear_pandas_nullable(n: int) -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    data = {"int": [1, 2, 3], "float": [0.1, 0.2, 0.3], "bool": [True, False, True]}
    df = nw.from_native(pd.DataFrame(data).convert_dtypes(dtype_backend="numpy_nullable"))

    df_clear = df.clear(n=n).lazy().collect()
    assert len(df_clear) == n
    assert df.collect_schema() == df_clear.collect_schema()

    assert_equal_data(df_clear, {k: [None] * n for k in data})
