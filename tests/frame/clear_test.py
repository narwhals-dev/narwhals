from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


@pytest.mark.parametrize("n", [0, 1, 10])
def test_clear(request: pytest.FixtureRequest, constructor: Constructor, n: int) -> None:
    data = {
        "int": [1, 2, 3],
        "str": ["foo", "bar", "baz"],
        "float": [0.1, 0.2, 0.3],
        "bool": [True, False, True],
    }
    df = nw.from_native(constructor(data))
    impl = df.implementation

    if n > 0 and (impl.is_pandas_like() or impl.is_dask()):
        reason = "NotImplementedError"
        request.applymarker(pytest.mark.xfail(reason))

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
