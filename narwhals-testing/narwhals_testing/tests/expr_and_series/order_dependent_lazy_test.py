from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from tests.utils import Constructor


def test_order_dependent_raises_in_lazy(constructor: Constructor) -> None:
    lf = nw.from_native(constructor({"a": [1, 2, 3]})).lazy()
    with pytest.raises(InvalidOperationError, match="Order-dependent expressions"):
        lf.select(nw.col("a").diff())
    with pytest.raises(InvalidOperationError, match="Order-dependent expressions"):
        lf.select(nw.sum_horizontal(nw.col("a").diff()))
    with pytest.raises(InvalidOperationError, match="Order-dependent expressions"):
        lf.select(nw.sum_horizontal(nw.col("a").diff(), nw.col("a")))

    for agg in ["max", "min", "mean", "sum", "median", "std", "var"]:
        with pytest.raises(InvalidOperationError, match="Order-dependent expressions"):
            lf.select(getattr(nw.col("a").diff(), agg)())
    for agg in ["any", "all"]:
        with pytest.raises(InvalidOperationError, match="Order-dependent expressions"):
            lf.select(getattr((nw.col("a").diff() > 0), agg)())
