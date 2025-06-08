from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._utils import Version

if TYPE_CHECKING:
    from narwhals.typing import IntoSeries
    from tests.utils import ConstructorEager


class MockCompliantSeries:
    _version = Version.MAIN

    def __narwhals_series__(self) -> Any:
        return self

    @property
    def native(self) -> tuple[()]:
        return ()

    @property
    def dtype(self) -> nw.Categorical:
        return nw.Categorical()


def test_is_ordered_categorical_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    s: IntoSeries | Any
    s = pl.Series(["a", "b"], dtype=pl.Categorical)
    assert nw.is_ordered_categorical(nw.from_native(s, series_only=True))
    s = pl.Series(["a", "b"], dtype=pl.Categorical(ordering="lexical"))
    assert not nw.is_ordered_categorical(nw.from_native(s, series_only=True))
    s = pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"]))
    assert nw.is_ordered_categorical(nw.from_native(s, series_only=True))


def test_is_ordered_categorical_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s = pd.Series(["a", "b"], dtype=pd.CategoricalDtype(ordered=True))
    assert nw.is_ordered_categorical(nw.from_native(s, series_only=True))
    s = pd.Series(["a", "b"], dtype=pd.CategoricalDtype(ordered=False))
    assert not nw.is_ordered_categorical(nw.from_native(s, series_only=True))


def test_is_ordered_categorical_pyarrow_string() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    tp = pa.dictionary(pa.int32(), pa.string())
    s = pa.chunked_array([pa.array(["a", "b"], type=tp)], type=tp)
    assert not nw.is_ordered_categorical(nw.from_native(s, series_only=True))


def test_is_definitely_not_ordered_categorical(
    constructor_eager: ConstructorEager,
) -> None:
    assert not nw.is_ordered_categorical(
        nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]
    )


@pytest.mark.xfail(reason="https://github.com/apache/arrow/issues/41017")
def test_is_ordered_categorical_pyarrow() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    tp = pa.dictionary(pa.int32(), pa.string(), ordered=True)
    arr = pa.array(["a", "b"], type=tp)
    s = pa.chunked_array([arr], type=tp)
    assert nw.is_ordered_categorical(
        nw.from_native(s, series_only=True)
    )  # pragma: no cover


def test_is_ordered_categorical_unknown_series() -> None:
    series: nw.Series[Any] = nw.Series(MockCompliantSeries(), level="full")
    assert nw.is_ordered_categorical(series) is False
