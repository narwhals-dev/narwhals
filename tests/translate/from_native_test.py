from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals as unstable_nw
import narwhals.stable.v1 as nw
from tests.utils import maybe_get_modin_df

data = {"a": [1, 2, 3]}

df_pd = pd.DataFrame(data)
df_pl = pl.DataFrame(data)
lf_pl = pl.LazyFrame(data)
df_mpd = maybe_get_modin_df(df_pd)
df_pa = pa.table(data)

series_pd = pd.Series(data["a"])
series_pl = pl.Series(data["a"])
series_mpd = df_mpd["a"]
series_pa = pa.chunked_array([data["a"]])


class MockDataFrame:
    def __narwhals_dataframe__(self) -> Any:
        return self


class MockLazyFrame:
    def __narwhals_lazyframe__(self) -> Any:
        return self


class MockSeries:
    def __narwhals_series__(self) -> Any:
        return self


eager_frames = [
    df_pd,
    df_pl,
    df_mpd,
    df_pa,
    MockDataFrame(),
]

lazy_frames = [
    lf_pl,
    MockLazyFrame(),
]

all_frames = [*eager_frames, *lazy_frames]

all_series = [
    series_pd,
    series_pl,
    series_mpd,
    series_pa,
    MockSeries(),
]


@pytest.mark.parametrize(
    ("strict", "context"),
    [
        (
            True,
            pytest.raises(
                TypeError,
                match="Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe",
            ),
        ),
        (False, does_not_raise()),
    ],
)
def test_strict(strict: Any, context: Any) -> None:
    arr = np.array([1, 2, 3])

    with context:
        res = nw.from_native(arr, strict=strict)
        assert isinstance(res, np.ndarray)


@pytest.mark.parametrize("dframe", lazy_frames)
@pytest.mark.parametrize(
    ("eager_only", "context"),
    [
        (False, does_not_raise()),
        (True, pytest.raises(TypeError, match="Cannot only use `eager_only`")),
    ],
)
def test_eager_only_lazy(dframe: Any, eager_only: Any, context: Any) -> None:
    with context:
        res = nw.from_native(dframe, eager_only=eager_only)
        assert isinstance(res, nw.LazyFrame)


@pytest.mark.parametrize("dframe", eager_frames)
@pytest.mark.parametrize("eager_only", [True, False])
def test_eager_only_eager(dframe: Any, eager_only: Any) -> None:
    res = nw.from_native(dframe, eager_only=eager_only)
    assert isinstance(res, nw.DataFrame)


@pytest.mark.parametrize(
    ("obj", "context"),
    [
        *[
            (frame, pytest.raises(TypeError, match="Cannot only use `series_only`"))
            for frame in all_frames
        ],
        *[(series, does_not_raise()) for series in all_series],
    ],
)
def test_series_only(obj: Any, context: Any) -> None:
    with context:
        res = nw.from_native(obj, series_only=True)
        assert isinstance(res, nw.Series)


@pytest.mark.parametrize("series", all_series)
@pytest.mark.parametrize(
    ("allow_series", "context"),
    [
        (True, does_not_raise()),
        (False, pytest.raises(TypeError, match="Please set `allow_series=True`")),
    ],
)
def test_allow_series(series: Any, allow_series: Any, context: Any) -> None:
    with context:
        res = nw.from_native(series, allow_series=allow_series)
        assert isinstance(res, nw.Series)


def test_invalid_series_combination() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid parameter combination: `series_only=True` and `allow_series=False`",
    ):
        nw.from_native(MockSeries(), series_only=True, allow_series=False)  # type: ignore[call-overload]


def test_pandas_like_validate() -> None:
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": [1, 2, 3]})
    df = pd.concat([df1, df2, df2], axis=1)

    with pytest.raises(
        ValueError, match=r"Expected unique column names, got:\n- 'b' 2 times"
    ):
        nw.from_native(df)


def test_init_already_narwhals() -> None:
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    result = nw.from_native(df)
    assert result is df  # type: ignore[comparison-overlap]
    s = df["a"]
    result_s = nw.from_native(s, allow_series=True)
    assert result_s is s


def test_init_already_narwhals_unstable() -> None:
    df = unstable_nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    result = unstable_nw.from_native(df)
    assert result is df  # type: ignore[comparison-overlap]
    s = df["a"]
    result_s = unstable_nw.from_native(s, allow_series=True)
    assert result_s is s


def test_series_only_dask() -> None:
    pytest.importorskip("dask")
    pytest.importorskip("dask_expr", exc_type=ImportError)
    import dask.dataframe as dd

    dframe = dd.from_pandas(df_pd)

    with pytest.raises(TypeError, match="Cannot only use `series_only`"):
        nw.from_native(dframe, series_only=True)


@pytest.mark.parametrize(
    ("eager_only", "context"),
    [
        (False, does_not_raise()),
        (True, pytest.raises(TypeError, match="Cannot only use `eager_only`")),
    ],
)
def test_eager_only_lazy_dask(eager_only: Any, context: Any) -> None:
    pytest.importorskip("dask")
    pytest.importorskip("dask_expr", exc_type=ImportError)
    import dask.dataframe as dd

    dframe = dd.from_pandas(df_pd)

    with context:
        res = nw.from_native(dframe, eager_only=eager_only)
        assert isinstance(res, nw.LazyFrame)


def test_from_native_strict_false_typing() -> None:
    df = pl.DataFrame()
    nw.from_native(df, strict=False)
    nw.from_native(df, strict=False, eager_only=True)
    nw.from_native(df, strict=False, eager_or_interchange_only=True)

    unstable_nw.from_native(df, strict=False)
    unstable_nw.from_native(df, strict=False, eager_only=True)
    unstable_nw.from_native(df, strict=False, eager_or_interchange_only=True)
