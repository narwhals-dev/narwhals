from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytest

import narwhals.stable.v1 as nw_v1
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from narwhals.typing import IntoSeries
    from tests.utils import ConstructorEager


def test_is_ordered_categorical_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    s: IntoSeries | Any
    s = pl.Series(["a", "b"], dtype=pl.Categorical)
    assert nw_v1.is_ordered_categorical(nw_v1.from_native(s, series_only=True))
    s = pl.Series(["a", "b"], dtype=pl.Categorical(ordering="lexical"))
    assert not nw_v1.is_ordered_categorical(nw_v1.from_native(s, series_only=True))
    s = pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"]))
    assert nw_v1.is_ordered_categorical(nw_v1.from_native(s, series_only=True))


def test_is_ordered_categorical_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s = pd.Series(["a", "b"], dtype=pd.CategoricalDtype(ordered=True))
    assert nw_v1.is_ordered_categorical(nw_v1.from_native(s, series_only=True))
    s = pd.Series(["a", "b"], dtype=pd.CategoricalDtype(ordered=False))
    assert not nw_v1.is_ordered_categorical(nw_v1.from_native(s, series_only=True))


def test_is_ordered_categorical_pyarrow_string() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    tp = pa.dictionary(pa.int32(), pa.string())
    s = pa.chunked_array([pa.array(["a", "b"], type=tp)], type=tp)
    assert not nw_v1.is_ordered_categorical(nw_v1.from_native(s, series_only=True))


@pytest.mark.skipif(PANDAS_VERSION < (2, 0), reason="requires interchange protocol")
def test_is_ordered_categorical_interchange_protocol() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = pd.DataFrame(
        {"a": ["a", "b"]}, dtype=pd.CategoricalDtype(ordered=True)
    ).__dataframe__()
    assert nw_v1.is_ordered_categorical(
        nw_v1.from_native(df, eager_or_interchange_only=True)["a"]
    )


def test_is_definitely_not_ordered_categorical(
    constructor_eager: ConstructorEager,
) -> None:
    assert not nw_v1.is_ordered_categorical(
        nw_v1.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]
    )


@pytest.mark.xfail(reason="https://github.com/apache/arrow/issues/41017")
def test_is_ordered_categorical_pyarrow() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    tp = pa.dictionary(pa.int32(), pa.string(), ordered=True)
    arr = pa.array(["a", "b"], type=tp)
    s = pa.chunked_array([arr], type=tp)
    assert nw_v1.is_ordered_categorical(
        nw_v1.from_native(s, series_only=True)
    )  # pragma: no cover
