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


def test_pandas_like_validate() -> None:
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": [1, 2, 3]})
    df = pd.concat([df1, df2, df2], axis=1)

    with pytest.raises(ValueError, match="Expected unique column names"):
        nw.from_native(df)


@pytest.mark.parametrize(
    ("series", "is_polars", "context"),
    [
        (MockSeries(), False, does_not_raise()),
        (MockSeries(), True, does_not_raise()),
        (series_pl, True, does_not_raise()),
        (
            series_pd,
            False,
            pytest.raises(
                TypeError,
                match="Expected Polars Series or an object which implements `__narwhals_series__`",
            ),
        ),
        (
            MockDataFrame(),
            False,
            pytest.raises(
                TypeError,
                match="Expected Polars Series or an object which implements `__narwhals_series__`",
            ),
        ),
    ],
)
def test_init_series(series: Any, is_polars: Any, context: Any) -> None:
    with context:
        result = nw.Series(
            series, is_polars=is_polars, backend_version=(1, 2, 3), level="full"
        )
        assert isinstance(result, nw.Series)


@pytest.mark.parametrize(
    ("dframe", "is_polars", "context"),
    [
        (MockDataFrame(), False, does_not_raise()),
        (MockDataFrame(), True, does_not_raise()),
        (df_pl, True, does_not_raise()),
        (
            df_pd,
            False,
            pytest.raises(
                TypeError,
                match="Expected Polars DataFrame or an object which implements `__narwhals_dataframe__`",
            ),
        ),
        (
            MockLazyFrame(),
            False,
            pytest.raises(
                TypeError,
                match="Expected Polars DataFrame or an object which implements `__narwhals_dataframe__`",
            ),
        ),
        (
            MockSeries(),
            True,
            pytest.raises(
                TypeError,
                match="Expected Polars DataFrame or an object which implements `__narwhals_dataframe__`",
            ),
        ),
    ],
)
def test_init_eager(dframe: Any, is_polars: Any, context: Any) -> None:
    with context:
        result = nw.DataFrame(
            dframe, is_polars=is_polars, backend_version=(1, 2, 3), level="full"
        )  # type: ignore[var-annotated]
        assert isinstance(result, nw.DataFrame)


@pytest.mark.parametrize(
    ("dframe", "is_polars", "context"),
    [
        (MockLazyFrame(), False, does_not_raise()),
        (MockLazyFrame(), True, does_not_raise()),
        (lf_pl, True, does_not_raise()),
        (
            df_pd,
            False,
            pytest.raises(
                TypeError,
                match="Expected Polars LazyFrame or an object that implements `__narwhals_lazyframe__`",
            ),
        ),
        (
            MockDataFrame(),
            False,
            pytest.raises(
                TypeError,
                match="Expected Polars LazyFrame or an object that implements `__narwhals_lazyframe__`",
            ),
        ),
        (
            MockSeries(),
            True,
            pytest.raises(
                TypeError,
                match="Expected Polars LazyFrame or an object that implements `__narwhals_lazyframe__`",
            ),
        ),
    ],
)
def test_init_lazy(dframe: Any, is_polars: Any, context: Any) -> None:
    with context:
        result = nw.LazyFrame(
            dframe, is_polars=is_polars, backend_version=(1, 2, 3), level="full"
        )  # type: ignore[var-annotated]
        assert isinstance(result, nw.LazyFrame)
