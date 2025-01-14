from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor


def test_order_dependent_raises_in_lazy(constructor: Constructor) -> None:
    lf = nw.from_native(constructor({"a": [1, 2, 3]})).lazy()
    with pytest.raises(TypeError, match="Order-dependent expressions"):
        lf.select(nw.col("a").diff())


def test_dask_order_dependent_ops() -> None:
    # Preserve these for narwhals.stable.v1, even though they
    # raise after stable.v1.
    pytest.importorskip("dask")
    pytest.importorskip("dask_expr", exc_type=ImportError)
    import dask.dataframe as dd

    df = nw_v1.from_native(dd.from_pandas(pd.DataFrame({"a": [1, 2, 3]})))
    result = df.select(
        a=nw.col("a").cum_sum(),
        b=nw.col("a").cum_count(),
        c=nw.col("a").cum_prod(),
        d=nw.col("a").cum_max(),
        e=nw.col("a").cum_min(),
        f=nw.col("a").shift(1),
        g=nw.col("a").diff(),
    )
    expected = {
        "a": [1, 3, 6],
        "b": [1, 2, 3],
        "c": [1, 2, 6],
        "d": [1, 2, 3],
        "e": [1, 1, 1],
        "f": [None, 1.0, 2.0],
        "g": [None, 1.0, 1.0],
    }
    assert_equal_data(result, expected)

    with pytest.raises(NotImplementedError):
        df.select(
            a=nw.col("a").cum_sum(reverse=True),
            b=nw.col("a").cum_count(reverse=True),
            c=nw.col("a").cum_prod(reverse=True),
            d=nw.col("a").cum_max(reverse=True),
            e=nw.col("a").cum_min(reverse=True),
            f=nw.col("a").shift(1),
            g=nw.col("a").diff(),
        )
