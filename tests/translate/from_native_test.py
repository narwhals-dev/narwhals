from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

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
    def __init__(self, *arg: Any, **kwargs: Any) -> None: ...
    def __narwhals_dataframe__(self) -> Any:
        return self


class MockLazyFrame:
    def __init__(self, *arg: Any, **kwargs: Any) -> None: ...
    def __narwhals_lazyframe__(self) -> Any:
        return self


class MockSeries:
    def __init__(self, *arg: Any, **kwargs: Any) -> None: ...
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


@pytest.mark.parametrize(
    ("dframe", "eager_only", "context"),
    [
        *[(lf, False, does_not_raise()) for lf in lazy_frames],
        *[
            (lf, True, pytest.raises(TypeError, match="Cannot only use `eager_only`"))
            for lf in lazy_frames
        ],
    ],
)
def test_eager_only_lazy(dframe: Any, eager_only: Any, context: Any) -> None:
    with context:
        res = nw.from_native(dframe, eager_only=eager_only)
        assert isinstance(res, nw.LazyFrame)


@pytest.mark.parametrize(
    ("dframe", "eager_only", "context"),
    [
        *[(df, False, does_not_raise()) for df in eager_frames],
        *[(df, True, does_not_raise()) for df in eager_frames],
    ],
)
def test_eager_only_eager(dframe: Any, eager_only: Any, context: Any) -> None:
    with context:
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


@pytest.mark.parametrize(
    ("series", "allow_series", "context"),
    [
        *[(series, True, does_not_raise()) for series in all_series],
        *[
            (
                series,
                False,
                pytest.raises(TypeError, match="Please set `allow_series=True`"),
            )
            for series in all_series
        ],
    ],
)
def test_allow_series(series: Any, allow_series: Any, context: Any) -> None:
    with context:
        res = nw.from_native(series, allow_series=allow_series)
        assert isinstance(res, nw.Series)
