from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw


def test_implementation_pandas() -> None:
    pd = pytest.importorskip("pandas")

    assert (
        nw.from_native(pd.DataFrame({"a": [1, 2, 3]})).implementation
        is nw.Implementation.PANDAS
    )
    assert (
        nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))["a"].implementation
        is nw.Implementation.PANDAS
    )
    assert nw.from_native(pd.DataFrame({"a": [1, 2, 3]})).implementation.is_pandas()
    assert nw.from_native(pd.DataFrame({"a": [1, 2, 3]})).implementation.is_pandas_like()


def test_implementation_polars() -> None:
    pl = pytest.importorskip("polars")

    assert not nw.from_native(pl.DataFrame({"a": [1, 2, 3]})).implementation.is_pandas()
    assert not nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))[
        "a"
    ].implementation.is_pandas()
    assert nw.from_native(pl.DataFrame({"a": [1, 2, 3]})).implementation.is_polars()
    assert nw.from_native(pl.LazyFrame({"a": [1, 2, 3]})).implementation.is_polars()
