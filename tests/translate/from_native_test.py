from __future__ import annotations

# mypy: disallow-any-generics=false, disable-error-code="var-annotated"
import sys
from contextlib import nullcontext as does_not_raise
from importlib.util import find_spec
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import cast

import numpy as np
import pytest

import narwhals as unstable_nw
import narwhals.stable.v1 as nw
from tests.utils import maybe_get_modin_df

if TYPE_CHECKING:
    from typing_extensions import Self
    from typing_extensions import assert_type

    from narwhals.utils import Version


class MockDataFrame:
    def _with_version(self: Self, _version: Version) -> MockDataFrame:
        return self

    def __narwhals_dataframe__(self: Self) -> Any:
        return self


class MockLazyFrame:
    def _with_version(self: Self, _version: Version) -> MockLazyFrame:
        return self

    def __narwhals_lazyframe__(self: Self) -> Any:
        return self


class MockSeries:
    def _with_version(self: Self, _version: Version) -> MockSeries:
        return self

    def __narwhals_series__(self: Self) -> Any:
        return self


data: dict[str, Any] = {"a": [1, 2, 3]}

eager_frames: list[Any] = [
    MockDataFrame(),
]
lazy_frames: list[Any] = [
    MockLazyFrame(),
]
all_series: list[Any] = [
    MockSeries(),
]

if find_spec("pandas") is not None:
    import pandas as pd

    df_pd: pd.DataFrame | None = pd.DataFrame(data)
    assert df_pd is not None
    df_mpd = maybe_get_modin_df(df_pd)
    series_pd = pd.Series(data["a"])
    series_mpd = df_mpd["a"]

    eager_frames += [
        df_pd,
        df_mpd,
    ]
    all_series += [
        series_pd,
        series_mpd,
    ]
else:  # pragma: no cover
    df_pd = None

if find_spec("polars") is not None:
    import polars as pl

    df_pl = pl.DataFrame(data)
    lf_pl: pl.LazyFrame | None = pl.LazyFrame(data)
    series_pl = pl.Series(data["a"])

    all_series += [
        series_pl,
    ]
    eager_frames += [
        df_pl,
    ]
    lazy_frames += [
        lf_pl,
    ]
else:  # pragma: no cover
    lf_pl = None

if find_spec("pyarrow") is not None:  # pragma: no cover
    import pyarrow as pa

    df_pa = pa.table(data)
    series_pa = pa.chunked_array([data["a"]])

    eager_frames += [
        df_pa,
    ]
    all_series += [
        series_pa,
    ]
else:  # pragma: no cover
    pass

all_frames = [*eager_frames, *lazy_frames]


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
    if eager_only:
        assert nw.from_native(dframe, eager_only=eager_only, strict=False) is dframe


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
    assert nw.from_native(obj, series_only=True, strict=False) is obj or isinstance(
        res, nw.Series
    )


@pytest.mark.parametrize("series", all_series)
@pytest.mark.parametrize(
    ("allow_series", "context"),
    [
        (True, does_not_raise()),
        (
            False,
            pytest.raises(
                TypeError, match="Please set `allow_series=True` or `series_only=True`"
            ),
        ),
    ],
)
def test_allow_series(series: Any, allow_series: Any, context: Any) -> None:
    with context:
        res = nw.from_native(series, allow_series=allow_series)
        assert isinstance(res, nw.Series)
    if not allow_series:
        assert nw.from_native(series, allow_series=allow_series, strict=False) is series


def test_invalid_series_combination() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid parameter combination: `series_only=True` and `allow_series=False`",
    ):
        nw.from_native(MockSeries(), series_only=True, allow_series=False)  # type: ignore[call-overload]


@pytest.mark.skipif(df_pd is None, reason="pandas not found")
def test_pandas_like_validate() -> None:
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": [1, 2, 3]})
    df = pd.concat([df1, df2, df2], axis=1)

    with pytest.raises(
        ValueError, match=r"Expected unique column names, got:\n- 'b' 2 times"
    ):
        nw.from_native(df)


@pytest.mark.skipif(lf_pl is None, reason="polars not found")
def test_init_already_narwhals() -> None:
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    result = nw.from_native(df)
    assert result is df
    s = df["a"]
    result_s = nw.from_native(s, allow_series=True)
    assert result_s is s


@pytest.mark.skipif(lf_pl is None, reason="polars not found")
def test_init_already_narwhals_unstable() -> None:
    df = unstable_nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    result = unstable_nw.from_native(df)
    assert result is df
    s = df["a"]
    result_s = unstable_nw.from_native(s, allow_series=True)
    assert result_s is s


@pytest.mark.skipif(df_pd is None, reason="pandas not found")
def test_series_only_dask() -> None:
    pytest.importorskip("dask")
    import dask.dataframe as dd

    dframe = dd.from_pandas(df_pd)

    with pytest.raises(TypeError, match="Cannot only use `series_only`"):
        nw.from_native(dframe, series_only=True)
    assert nw.from_native(dframe, series_only=True, strict=False) is dframe


@pytest.mark.skipif(df_pd is None, reason="pandas not found")
@pytest.mark.parametrize(
    ("eager_only", "context"),
    [
        (False, does_not_raise()),
        (True, pytest.raises(TypeError, match="Cannot only use `eager_only`")),
    ],
)
def test_eager_only_lazy_dask(eager_only: Any, context: Any) -> None:
    pytest.importorskip("dask")
    import dask.dataframe as dd

    dframe = dd.from_pandas(df_pd)

    with context:
        res = nw.from_native(dframe, eager_only=eager_only)
        assert isinstance(res, nw.LazyFrame)
    if eager_only:
        assert nw.from_native(dframe, eager_only=eager_only, strict=False) is dframe


@pytest.mark.skipif(sys.version_info < (3, 9), reason="too old for sqlframe")
def test_series_only_sqlframe() -> None:  # pragma: no cover
    pytest.importorskip("sqlframe")
    from sqlframe.duckdb import DuckDBSession

    session = DuckDBSession()
    df = session.createDataFrame([*zip(*data.values())], schema=[*data.keys()])

    with pytest.raises(TypeError, match="Cannot only use `series_only`"):
        nw.from_native(df, series_only=True)  # pyright: ignore[reportArgumentType, reportCallIssue]


@pytest.mark.parametrize(
    ("eager_only", "context"),
    [
        (False, does_not_raise()),
        (True, pytest.raises(TypeError, match="Cannot only use `eager_only`")),
    ],
)
@pytest.mark.skipif(sys.version_info < (3, 9), reason="too old for sqlframe")
def test_eager_only_sqlframe(eager_only: Any, context: Any) -> None:  # pragma: no cover
    pytest.importorskip("sqlframe")
    from sqlframe.duckdb import DuckDBSession

    session = DuckDBSession()
    df = session.createDataFrame([*zip(*data.values())], schema=[*data.keys()])

    with context:
        res = nw.from_native(df, eager_only=eager_only)
        assert isinstance(res, nw.LazyFrame)


@pytest.mark.skipif(lf_pl is None, reason="polars not found")
def test_from_native_strict_false_typing() -> None:
    df = pl.DataFrame()
    nw.from_native(df, strict=False)
    nw.from_native(df, strict=False, eager_only=True)
    nw.from_native(df, strict=False, eager_or_interchange_only=True)

    with pytest.deprecated_call(match="please use `pass_through` instead"):
        unstable_nw.from_native(df, strict=False)  # type: ignore[call-overload]
        unstable_nw.from_native(df, strict=False, eager_only=True)  # type: ignore[call-overload]


def test_from_native_strict_false_invalid() -> None:
    with pytest.raises(ValueError, match="Cannot pass both `strict`"):
        nw.from_native({"a": [1, 2, 3]}, strict=True, pass_through=False)  # type: ignore[call-overload]


def test_from_mock_interchange_protocol_non_strict() -> None:
    class MockDf:
        def __dataframe__(self) -> None:  # pragma: no cover
            pass

    mockdf = MockDf()
    result = nw.from_native(mockdf, eager_only=True, strict=False)
    assert result is mockdf


def test_from_native_strict_native_series() -> None:
    obj: list[int] = [1, 2, 3, 4]
    array_like = cast("Iterable[Any]", obj)
    not_array_like: Literal[1] = 1

    with pytest.raises(TypeError, match="got.+list"):
        nw.from_native(obj, series_only=True)  # type: ignore[call-overload]

    with pytest.raises(TypeError, match="got.+list"):
        nw.from_native(array_like, series_only=True)  # type: ignore[call-overload]

    with pytest.raises(TypeError, match="got.+int"):
        nw.from_native(not_array_like, series_only=True)  # type: ignore[call-overload]


@pytest.mark.skipif(lf_pl is None, reason="polars not found")
def test_from_native_strict_native_series_polars() -> None:
    obj: list[int] = [1, 2, 3, 4]
    np_array = pl.Series(obj).to_numpy()
    with pytest.raises(TypeError, match="got.+numpy.ndarray"):
        nw.from_native(np_array, series_only=True)  # type: ignore[call-overload]


@pytest.mark.skipif(lf_pl is None, reason="polars not found")
def test_from_native_lazyframe() -> None:
    assert lf_pl is not None
    stable_lazy = nw.from_native(lf_pl)
    unstable_lazy = unstable_nw.from_native(lf_pl)
    if TYPE_CHECKING:
        assert_type(stable_lazy, nw.LazyFrame[pl.LazyFrame])
        assert_type(unstable_lazy, unstable_nw.LazyFrame[pl.LazyFrame])

    assert isinstance(stable_lazy, nw.LazyFrame)
    assert isinstance(unstable_lazy, unstable_nw.LazyFrame)


def test_dataframe_recursive() -> None:
    pytest.importorskip("polars")
    import polars as pl

    pl_frame = pl.DataFrame({"a": [1, 2, 3]})
    nw_frame = unstable_nw.from_native(pl_frame)
    with pytest.raises(AssertionError):
        unstable_nw.DataFrame(nw_frame, level="full")

    nw_frame_early_return = unstable_nw.from_native(nw_frame)

    if TYPE_CHECKING:
        assert_type(pl_frame, pl.DataFrame)
        assert_type(nw_frame, unstable_nw.DataFrame[pl.DataFrame])

        nw_frame_depth_2 = unstable_nw.DataFrame(nw_frame, level="full")
        # NOTE: Checking that the type is `DataFrame[Unknown]`
        assert_type(nw_frame_depth_2, unstable_nw.DataFrame)
        assert_type(nw_frame_early_return, unstable_nw.DataFrame[pl.DataFrame])


def test_lazyframe_recursive() -> None:
    pytest.importorskip("polars")
    import polars as pl

    pl_frame = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    nw_frame = unstable_nw.from_native(pl_frame)
    with pytest.raises(AssertionError):
        unstable_nw.LazyFrame(nw_frame, level="lazy")

    nw_frame_early_return = unstable_nw.from_native(nw_frame)

    if TYPE_CHECKING:
        assert_type(pl_frame, pl.LazyFrame)
        assert_type(nw_frame, unstable_nw.LazyFrame[pl.LazyFrame])

        nw_frame_depth_2 = unstable_nw.LazyFrame(nw_frame, level="lazy")
        # NOTE: Checking that the type is `LazyFrame[Unknown]`
        assert_type(nw_frame_depth_2, unstable_nw.LazyFrame)
        assert_type(nw_frame_early_return, unstable_nw.LazyFrame[pl.LazyFrame])


def test_dataframe_recursive_v1() -> None:
    """`v1` always returns a Union."""
    pytest.importorskip("polars")
    import polars as pl

    pl_frame = pl.DataFrame({"a": [1, 2, 3]})
    nw_frame = nw.from_native(pl_frame)
    with pytest.raises(AssertionError):
        nw.DataFrame(nw_frame, level="full")

    nw_frame_early_return = nw.from_native(nw_frame)

    if TYPE_CHECKING:
        assert_type(pl_frame, pl.DataFrame)
        # TODO @dangotbanned: Fix without breaking something else (1)
        assert_type(nw_frame, "nw.DataFrame[pl.DataFrame] | nw.LazyFrame[pl.DataFrame]")

        nw_frame_depth_2 = nw.DataFrame(nw_frame, level="full")
        # NOTE: Checking that the type is `DataFrame[Unknown]`
        assert_type(nw_frame_depth_2, nw.DataFrame)
        # TODO @dangotbanned: Fix without breaking something else (2)
        assert_type(
            nw_frame_early_return,
            "nw.DataFrame[pl.DataFrame] | nw.LazyFrame[pl.DataFrame]",
        )


def test_lazyframe_recursive_v1() -> None:
    pytest.importorskip("polars")
    import polars as pl

    pl_frame = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    nw_frame = nw.from_native(pl_frame)
    with pytest.raises(AssertionError):
        nw.LazyFrame(nw_frame, level="lazy")

    nw_frame_early_return = nw.from_native(nw_frame)

    if TYPE_CHECKING:
        assert_type(pl_frame, pl.LazyFrame)
        assert_type(nw_frame, nw.LazyFrame[pl.LazyFrame])

        nw_frame_depth_2 = nw.LazyFrame(nw_frame, level="lazy")
        # NOTE: Checking that the type is `LazyFrame[Unknown]`
        assert_type(nw_frame_depth_2, nw.LazyFrame)
        assert_type(nw_frame_early_return, nw.LazyFrame[pl.LazyFrame])


def test_series_recursive() -> None:
    """https://github.com/narwhals-dev/narwhals/issues/2239."""
    pytest.importorskip("polars")
    import polars as pl

    pl_series = pl.Series(name="test", values=[1, 2, 3])
    nw_series = unstable_nw.from_native(pl_series, series_only=True)
    with pytest.raises(AssertionError):
        unstable_nw.Series(nw_series, level="full")

    nw_series_early_return = unstable_nw.from_native(nw_series, series_only=True)

    if TYPE_CHECKING:
        assert_type(pl_series, pl.Series)
        assert_type(nw_series, unstable_nw.Series[pl.Series])

        nw_series_depth_2 = unstable_nw.Series(nw_series, level="full")
        # NOTE: Checking that the type is `Series[Unknown]`
        assert_type(nw_series_depth_2, unstable_nw.Series)
        assert_type(nw_series_early_return, unstable_nw.Series[pl.Series])


def test_series_recursive_v1() -> None:
    """https://github.com/narwhals-dev/narwhals/issues/2239."""
    pytest.importorskip("polars")
    import polars as pl

    pl_series = pl.Series(name="test", values=[1, 2, 3])
    nw_series = nw.from_native(pl_series, series_only=True)
    with pytest.raises(AssertionError):
        nw.Series(nw_series, level="full")

    nw_series_early_return = nw.from_native(nw_series, series_only=True)

    if TYPE_CHECKING:
        assert_type(pl_series, pl.Series)
        assert_type(nw_series, nw.Series[pl.Series])

        nw_series_depth_2 = nw.Series(nw_series, level="full")
        # NOTE: `Unknown` isn't possible for `v1`, as it has a `TypeVar` default
        assert_type(nw_series_depth_2, nw.Series[Any])
        assert_type(nw_series_early_return, nw.Series[pl.Series])


@pytest.mark.parametrize("from_native", [unstable_nw.from_native, nw.from_native])
def test_from_native_invalid_keywords(from_native: Callable[..., Any]) -> None:
    pattern = r"from_native.+unexpected.+keyword.+bad_1"

    with pytest.raises(TypeError, match=pattern):
        from_native(data, bad_1="invalid")

    with pytest.raises(TypeError, match=pattern):
        from_native(data, bad_1="invalid", bad_2="also invalid")
