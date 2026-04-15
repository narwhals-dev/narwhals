from __future__ import annotations

import narwhals as nw
from narwhals.testing.constructors import ConstructorName
from narwhals.testing.constructors._classes import DaskConstructor


def test_dask_npartitions_distinct() -> None:
    dp1, dp2 = DaskConstructor(npartitions=1), DaskConstructor(npartitions=2)
    assert dp1 != dp2
    assert hash(dp1) != hash(dp2)


def test_dask_repr() -> None:
    assert repr(DaskConstructor(npartitions=3)) == "DaskConstructor(npartitions=3)"


def test_eager_returns_eager_frame() -> None:
    data = {"x": [1, 2, 3]}
    constructor = ConstructorName.PANDAS.constructor
    df = nw.from_native(constructor(data))
    assert isinstance(df, nw.DataFrame)


def test_lazy_returns_lazy_frame() -> None:
    data = {"x": [1, 2, 3]}
    constructor = ConstructorName.POLARS_LAZY.constructor
    lf = nw.from_native(constructor(data))
    assert isinstance(lf, nw.LazyFrame)
