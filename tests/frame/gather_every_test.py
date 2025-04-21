from __future__ import annotations

import pandas as pd
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": list(range(10))}


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_gather_every(constructor_eager: ConstructorEager, n: int, offset: int) -> None:
    df_v1 = nw_v1.from_native(constructor_eager(data))
    result = df_v1.gather_every(n=n, offset=offset)
    expected = {"a": data["a"][offset::n]}
    assert_equal_data(result, expected)

    # Test deprecation for LazyFrame in main namespace
    lf = nw.from_native(constructor_eager(data)).lazy()
    with pytest.deprecated_call(
        match="is deprecated and will be removed in a future version"
    ):
        lf.gather_every(n=n, offset=offset)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_gather_every_dask_v1(n: int, offset: int) -> None:
    pytest.importorskip("dask")
    import dask.dataframe as dd

    df_v1 = nw_v1.from_native(dd.from_pandas(pd.DataFrame(data)))
    result = df_v1.gather_every(n=n, offset=offset)
    expected = {"a": data["a"][offset::n]}
    assert_equal_data(result, expected)
