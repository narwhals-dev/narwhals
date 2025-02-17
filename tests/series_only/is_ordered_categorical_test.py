from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from narwhals.typing import IntoSeries
    from tests.utils import ConstructorEager


def test_is_ordered_categorical() -> None:
    s: IntoSeries | Any
    s = pl.Series(["a", "b"], dtype=pl.Categorical)
    assert nw.is_ordered_categorical(nw.from_native(s, series_only=True))
    s = pl.Series(["a", "b"], dtype=pl.Categorical(ordering="lexical"))
    assert not nw.is_ordered_categorical(nw.from_native(s, series_only=True))
    s = pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"]))
    assert nw.is_ordered_categorical(nw.from_native(s, series_only=True))
    s = pd.Series(["a", "b"], dtype=pd.CategoricalDtype(ordered=True))
    assert nw.is_ordered_categorical(nw.from_native(s, series_only=True))
    s = pd.Series(["a", "b"], dtype=pd.CategoricalDtype(ordered=False))
    assert not nw.is_ordered_categorical(nw.from_native(s, series_only=True))
    tp = pa.dictionary(pa.int32(), pa.string())
    s = pa.chunked_array([pa.array(["a", "b"], type=tp)], type=tp)
    assert not nw.is_ordered_categorical(nw.from_native(s, series_only=True))


@pytest.mark.skipif(PANDAS_VERSION < (2, 0), reason="requires interchange protocol")
def test_is_ordered_categorical_interchange_protocol() -> None:
    df = pd.DataFrame(
        {"a": ["a", "b"]}, dtype=pd.CategoricalDtype(ordered=True)
    ).__dataframe__()
    assert nw.is_ordered_categorical(
        nw.from_native(df, eager_or_interchange_only=True)["a"]
    )


def test_is_definitely_not_ordered_categorical(
    constructor_eager: ConstructorEager,
) -> None:
    assert not nw.is_ordered_categorical(
        nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]
    )


@pytest.mark.xfail(reason="https://github.com/apache/arrow/issues/41017")
def test_is_ordered_categorical_pyarrow() -> None:
    tp = pa.dictionary(pa.int32(), pa.string(), ordered=True)
    s = pa.chunked_array([pa.array(["a", "b"], type=tp)])  # type: ignore[list-item]
    assert nw.is_ordered_categorical(nw.from_native(s, series_only=True))
