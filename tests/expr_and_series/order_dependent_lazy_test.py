from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.exceptions import OrderDependentExprError
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor


def test_order_dependent_raises_in_lazy(constructor: Constructor) -> None:
    lf = nw.from_native(constructor({"a": [1, 2, 3]})).lazy()
    with pytest.raises(OrderDependentExprError, match="Order-dependent expressions"):
        lf.select(nw.col("a").diff())
    with pytest.raises(OrderDependentExprError, match="Order-dependent expressions"):
        lf.select(nw.sum_horizontal(nw.col("a").diff()))
    with pytest.raises(OrderDependentExprError, match="Order-dependent expressions"):
        lf.select(nw.sum_horizontal(nw.col("a").diff(), nw.col("a")))

    for agg in ["max", "min", "mean", "sum", "median", "std", "var"]:
        with pytest.raises(OrderDependentExprError, match="Order-dependent expressions"):
            lf.select(getattr(nw.col("a").diff(), agg)())
    for agg in ["any", "all"]:
        with pytest.raises(OrderDependentExprError, match="Order-dependent expressions"):
            lf.select(getattr((nw.col("a").diff() > 0), agg)())


def test_dask_order_dependent_ops() -> None:
    # Preserve these for narwhals.stable.v1, even though they
    # raise after stable.v1.
    pytest.importorskip("dask")
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
        h=nw.col("a").is_first_distinct(),
        i=nw.col("a").is_last_distinct(),
    )
    expected = {
        "a": [1, 3, 6],
        "b": [1, 2, 3],
        "c": [1, 2, 6],
        "d": [1, 2, 3],
        "e": [1, 1, 1],
        "f": [None, 1.0, 2.0],
        "g": [None, 1.0, 1.0],
        "h": [True, True, True],
        "i": [True, True, True],
    }
    assert_equal_data(result, expected)
