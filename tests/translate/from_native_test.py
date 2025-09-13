"""`from_native` runtime and static typing tests.

# Static Typing
The style of the tests is *intentionally* repetitive, aiming to provide an individual scope
for each attempted `@overload` match.

## `mypy` ignores
[inline config] is used to prevent [mypy specific errors] from hiding `pyright` diagnostics.

[`--disallow-any-generics`] and [`var-annotated`] are ignored to verify we don't regress to
**prior false positive** behaviors identified in [#2239].

[inline config]: https://mypy.readthedocs.io/en/stable/inline_config.html
[mypy specific errors]: https://discuss.python.org/t/ignore-mypy-specific-type-errors/58535
[`--disallow-any-generics`]: https://mypy.readthedocs.io/en/stable/error_code_list2.html#check-that-type-arguments-exist-type-arg
[`var-annotated`]: https://mypy.readthedocs.io/en/stable/error_code_list.html#require-annotation-if-variable-type-is-unclear-var-annotated
[#2239]: https://github.com/narwhals-dev/narwhals/issues/2239
"""

from __future__ import annotations

# Using pyright's assert type instead
# mypy: disallow-any-generics=false, disable-error-code="assert-type"
from contextlib import nullcontext as does_not_raise
from importlib.util import find_spec
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, cast

import pytest

import narwhals as nw
from narwhals._utils import Version
from tests.conftest import sqlframe_pyspark_lazy_constructor
from tests.utils import Constructor, maybe_get_modin_df

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from _pytest.mark import ParameterSet
    from typing_extensions import assert_type


class MockDataFrame:
    def __init__(self, version: Version) -> None:
        self._version = version

    def _with_version(self, version: Version) -> MockDataFrame:
        return self.__class__(version)

    def __narwhals_dataframe__(self) -> Any:
        return self


class MockLazyFrame:
    def __init__(self, version: Version) -> None:
        self._version = version

    def _with_version(self, version: Version) -> MockLazyFrame:
        return self.__class__(version)

    def __narwhals_lazyframe__(self) -> Any:
        return self


class MockSeries:
    def __init__(self, version: Version) -> None:
        self._version = version

    def _with_version(self, version: Version) -> MockSeries:
        return self.__class__(version)

    def __narwhals_series__(self) -> Any:
        return self


data: dict[str, Any] = {"a": [1, 2, 3]}

eager_frames: list[Any] = [MockDataFrame(Version.MAIN)]
lazy_frames: list[Any] = [MockLazyFrame(Version.MAIN)]
all_series: list[Any] = [MockSeries(Version.MAIN)]

if find_spec("pandas") is not None:
    import pandas as pd

    df_pd: pd.DataFrame | None = pd.DataFrame(data)
    assert df_pd is not None
    df_mpd = maybe_get_modin_df(df_pd)
    series_pd = pd.Series(data["a"])
    series_mpd = df_mpd["a"]

    eager_frames += [df_pd, df_mpd]
    all_series += [series_pd, series_mpd]
else:  # pragma: no cover
    df_pd = None

if find_spec("polars") is not None:
    import polars as pl

    df_pl = pl.DataFrame(data)
    lf_pl: pl.LazyFrame | None = pl.LazyFrame(data)
    series_pl = pl.Series(data["a"])

    all_series += [series_pl]
    eager_frames += [df_pl]
    lazy_frames += [lf_pl]
else:  # pragma: no cover
    lf_pl = None

if find_spec("pyarrow") is not None:  # pragma: no cover
    import pyarrow as pa

    df_pa = pa.table(data)
    series_pa = pa.chunked_array([data["a"]])

    eager_frames += [df_pa]
    all_series += [series_pa]
else:  # pragma: no cover
    pass

all_frames = [*eager_frames, *lazy_frames]


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
        assert nw.from_native(dframe, eager_only=eager_only, pass_through=True) is dframe


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
    assert nw.from_native(obj, series_only=True, pass_through=True) is obj or isinstance(
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
        assert (
            nw.from_native(series, allow_series=allow_series, pass_through=True) is series
        )


def test_invalid_series_combination() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid parameter combination: `series_only=True` and `allow_series=False`",
    ):
        nw.from_native(MockSeries(Version.V1), series_only=True, allow_series=False)  # type: ignore[call-overload]


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
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    result = nw.from_native(df)
    assert result is df
    s = df["a"]
    result_s = nw.from_native(s, allow_series=True)
    assert result_s is s


@pytest.mark.skipif(df_pd is None, reason="pandas not found")
def test_series_only_dask() -> None:
    pytest.importorskip("dask")
    import dask.dataframe as dd

    dframe = dd.from_pandas(df_pd)

    with pytest.raises(TypeError, match="Cannot only use `series_only`"):
        nw.from_native(dframe, series_only=True)
    assert nw.from_native(dframe, series_only=True, pass_through=True) is dframe


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
        assert nw.from_native(dframe, eager_only=eager_only, pass_through=True) is dframe


def test_series_only_sqlframe() -> None:  # pragma: no cover
    pytest.importorskip("sqlframe")
    df = sqlframe_pyspark_lazy_constructor(data)

    with pytest.raises(TypeError, match="Cannot only use `series_only`"):
        nw.from_native(df, series_only=True)  # pyright: ignore[reportArgumentType, reportCallIssue]


@pytest.mark.parametrize(
    ("eager_only", "context"),
    [
        (False, does_not_raise()),
        (
            True,
            pytest.raises(
                TypeError,
                match="Cannot only use `series_only`, `eager_only` or `eager_or_interchange_only` with sqlframe DataFrame",
            ),
        ),
    ],
)
def test_eager_only_sqlframe(eager_only: Any, context: Any) -> None:  # pragma: no cover
    pytest.importorskip("sqlframe")
    df = sqlframe_pyspark_lazy_constructor(data)

    with context:
        res = nw.from_native(df, eager_only=eager_only)
        assert isinstance(res, nw.LazyFrame)


def test_interchange_protocol_non_v1() -> None:
    class MockDf:
        def __dataframe__(self) -> None:  # pragma: no cover
            pass

    mockdf = MockDf()
    result = nw.from_native(mockdf, pass_through=True)
    assert result is mockdf
    with pytest.raises(TypeError):
        nw.from_native(mockdf)  # type: ignore[call-overload]


def test_from_native_strict_native_series() -> None:
    obj: list[int] = [1, 2, 3, 4]
    array_like = cast("Iterable[Any]", obj)
    not_array_like: Literal[1] = 1

    with pytest.raises(TypeError, match=r"got.+list"):
        nw.from_native(obj, series_only=True)  # type: ignore[call-overload]

    with pytest.raises(TypeError, match=r"got.+list"):
        nw.from_native(array_like, series_only=True)  # type: ignore[call-overload]

    with pytest.raises(TypeError, match=r"got.+int"):
        nw.from_native(not_array_like, series_only=True)  # type: ignore[call-overload]


@pytest.mark.skipif(lf_pl is None, reason="polars not found")
def test_from_native_strict_native_series_polars() -> None:
    obj: list[int] = [1, 2, 3, 4]
    np_array = pl.Series(obj).to_numpy()
    with pytest.raises(TypeError, match=r"got.+numpy.ndarray"):
        nw.from_native(np_array, series_only=True)  # type: ignore[call-overload]


def test_dataframe_recursive() -> None:
    pytest.importorskip("polars")
    import polars as pl

    pl_frame = pl.DataFrame({"a": [1, 2, 3]})
    nw_frame = nw.from_native(pl_frame)
    with pytest.raises(AssertionError):
        nw.DataFrame(nw_frame, level="full")

    nw_frame_early_return = nw.from_native(nw_frame)

    if TYPE_CHECKING:
        assert_type(pl_frame, pl.DataFrame)
        assert_type(nw_frame, nw.DataFrame[pl.DataFrame])

        nw_frame_depth_2 = nw.DataFrame(nw_frame, level="full")  # type: ignore[var-annotated]
        # NOTE: Checking that the type is `DataFrame[Unknown]`
        assert_type(nw_frame_depth_2, nw.DataFrame[Any])
        assert_type(nw_frame_early_return, nw.DataFrame[pl.DataFrame])


def test_lazyframe_recursive() -> None:
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

        nw_frame_depth_2 = nw.LazyFrame(nw_frame, level="lazy")  # type: ignore[var-annotated]
        # NOTE: Checking that the type is `LazyFrame[Unknown]`
        assert_type(nw_frame_depth_2, nw.LazyFrame[Any])
        assert_type(nw_frame_early_return, nw.LazyFrame[pl.LazyFrame])


def test_series_recursive() -> None:
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

        nw_series_depth_2 = nw.Series(nw_series, level="full")  # type: ignore[var-annotated]
        # NOTE: Checking that the type is `Series[Unknown]`
        assert_type(nw_series_depth_2, nw.Series[Any])
        assert_type(nw_series_early_return, nw.Series[pl.Series])


def test_from_native_invalid_keywords() -> None:
    pattern = r"from_native.+unexpected.+keyword.+bad_1"

    with pytest.raises(TypeError, match=pattern):
        nw.from_native(data, bad_1="invalid")  # type: ignore[call-overload]

    with pytest.raises(TypeError, match=pattern):
        nw.from_native(data, bad_1="invalid", bad_2="also invalid")  # type: ignore[call-overload]


def _iter_roundtrip_cases(iterable: Iterable[Any], **kwds: Any) -> Iterator[ParameterSet]:
    for element in iterable:
        tp = type(element)
        if not tp.__name__.startswith("Mock"):
            yield pytest.param(element, kwds, id=f"{tp.__module__}.{tp.__qualname__}")


@pytest.mark.parametrize(
    ("native", "kwds"),
    list(
        chain(
            _iter_roundtrip_cases(all_frames),
            _iter_roundtrip_cases(all_series, allow_series=True),
        )
    ),
)
def test_from_native_roundtrip_identity(native: Any, kwds: dict[str, Any]) -> None:
    nw_wrapped = nw.from_native(native, **kwds)
    roundtrip = nw_wrapped.to_native()
    assert roundtrip is native


def test_pyspark_connect_deps_2517(constructor: Constructor) -> None:  # pragma: no cover
    if not ("pyspark" in str(constructor) and "sqlframe" not in str(constructor)):
        # Only run this slow test if `--constructors=pyspark` is passed
        return
    pytest.importorskip("pyspark")
    # Don't delete this! It's crucial for the test that
    # pyspark.sql.connect be imported.
    import pyspark.sql.connect  # noqa: F401
    from pyspark.sql import SparkSession

    import narwhals as nw

    spark = SparkSession.builder.getOrCreate()
    # Check this doesn't raise
    nw.from_native(spark.createDataFrame([(1,)], ["a"]))


def test_eager_only_pass_through_main(constructor: Constructor) -> None:
    if not any(s in str(constructor) for s in ("pyspark", "dask", "ibis", "duckdb")):
        pytest.skip(reason="Non lazy or polars")

    df = constructor(data)

    r1 = nw.from_native(df, eager_only=False, pass_through=False)
    r2 = nw.from_native(df, eager_only=False, pass_through=True)
    r3 = nw.from_native(df, eager_only=True, pass_through=True)

    assert isinstance(r1, nw.LazyFrame)
    assert isinstance(r2, nw.LazyFrame)
    assert not isinstance(r3, nw.LazyFrame)

    with pytest.raises(TypeError, match=r"Cannot.+use.+eager_only"):
        nw.from_native(df, eager_only=True, pass_through=False)  # type: ignore[type-var]


def test_from_native_lazyframe_exhaustive() -> None:  # noqa: PLR0914, PLR0915
    pytest.importorskip("polars")
    pytest.importorskip("typing_extensions")

    import polars as pl
    from typing_extensions import assert_type

    pl_ldf = pl.LazyFrame(data)

    pl_1 = nw.from_native(pl_ldf)
    pl_2 = nw.from_native(pl_ldf, pass_through=False)
    pl_3 = nw.from_native(pl_ldf, pass_through=True)
    pl_4 = nw.from_native(pl_ldf, eager_only=False)
    pl_5 = nw.from_native(pl_ldf, series_only=False)
    pl_6 = nw.from_native(pl_ldf, allow_series=False)
    pl_7 = nw.from_native(pl_ldf, allow_series=None)
    pl_8 = nw.from_native(pl_ldf, allow_series=True)
    pl_9 = nw.from_native(pl_ldf, pass_through=False, eager_only=False)
    pl_10 = nw.from_native(pl_ldf, pass_through=True, eager_only=False)
    pl_11 = nw.from_native(
        pl_ldf, pass_through=False, eager_only=False, series_only=False
    )
    pl_12 = nw.from_native(pl_ldf, pass_through=True, eager_only=False, series_only=False)
    pl_13 = nw.from_native(
        pl_ldf, pass_through=False, eager_only=False, allow_series=False
    )
    pl_14 = nw.from_native(
        pl_ldf, pass_through=True, eager_only=False, allow_series=False
    )
    pl_15 = nw.from_native(
        pl_ldf,
        pass_through=False,
        eager_only=False,
        series_only=False,
        allow_series=False,
    )
    pl_16 = nw.from_native(
        pl_ldf, pass_through=True, eager_only=False, series_only=False, allow_series=False
    )
    pl_17 = nw.from_native(
        pl_ldf, pass_through=False, eager_only=False, allow_series=None
    )
    pl_18 = nw.from_native(pl_ldf, pass_through=True, eager_only=False, allow_series=None)
    pl_19 = nw.from_native(
        pl_ldf, pass_through=False, eager_only=False, series_only=False, allow_series=None
    )
    pl_20 = nw.from_native(
        pl_ldf, pass_through=True, eager_only=False, series_only=False, allow_series=None
    )
    pl_21 = nw.from_native(
        pl_ldf, pass_through=False, eager_only=False, allow_series=True
    )
    pl_21 = nw.from_native(pl_ldf, pass_through=True, eager_only=False, allow_series=True)
    pl_21 = nw.from_native(
        pl_ldf, pass_through=False, eager_only=False, series_only=False, allow_series=True
    )
    pl_22 = nw.from_native(
        pl_ldf, pass_through=True, eager_only=False, series_only=False, allow_series=True
    )
    pl_23 = nw.from_native(pl_ldf, eager_only=False, series_only=False)
    pl_24 = nw.from_native(pl_ldf, eager_only=False, allow_series=False)
    pl_25 = nw.from_native(
        pl_ldf, eager_only=False, series_only=False, allow_series=False
    )
    pl_26 = nw.from_native(pl_ldf, eager_only=False, allow_series=None)
    pl_27 = nw.from_native(pl_ldf, eager_only=False, series_only=False, allow_series=None)
    pl_28 = nw.from_native(pl_ldf, eager_only=False, allow_series=True)
    pl_29 = nw.from_native(pl_ldf, eager_only=False, series_only=False, allow_series=True)
    pl_30 = nw.from_native(
        pl_ldf, pass_through=False, series_only=False, allow_series=None
    )
    pl_31 = nw.from_native(
        pl_ldf, pass_through=False, series_only=False, allow_series=False
    )
    pl_32 = nw.from_native(
        pl_ldf, pass_through=False, series_only=False, allow_series=True
    )
    pl_33 = nw.from_native(
        pl_ldf, pass_through=True, series_only=False, allow_series=None
    )
    pl_34 = nw.from_native(
        pl_ldf, pass_through=True, series_only=False, allow_series=False
    )
    pl_35 = nw.from_native(
        pl_ldf, pass_through=True, series_only=False, allow_series=True
    )
    pls = (
        pl_1,
        pl_2,
        pl_3,
        pl_4,
        pl_5,
        pl_6,
        pl_7,
        pl_8,
        pl_9,
        pl_10,
        pl_11,
        pl_12,
        pl_13,
        pl_14,
        pl_15,
        pl_16,
        pl_17,
        pl_18,
        pl_19,
        pl_20,
        pl_21,
        pl_22,
        pl_23,
        pl_24,
        pl_25,
        pl_26,
        pl_27,
        pl_28,
        pl_29,
        pl_30,
        pl_31,
        pl_32,
        pl_33,
        pl_34,
        pl_35,
    )

    assert_type(pl_1, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_2, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_3, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_4, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_5, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_6, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_7, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_8, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_9, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_10, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_11, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_12, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_13, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_14, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_15, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_16, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_17, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_18, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_19, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_20, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_21, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_22, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_23, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_24, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_25, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_26, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_27, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_28, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_29, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_30, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_31, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_32, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_33, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_34, nw.LazyFrame[pl.LazyFrame])
    assert_type(pl_35, nw.LazyFrame[pl.LazyFrame])

    for ldf in pls:
        assert isinstance(ldf, nw.LazyFrame)


def test_from_native_series_exhaustive() -> None:  # noqa: PLR0914, PLR0915
    pytest.importorskip("polars")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("typing_extensions")
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from typing_extensions import assert_type

    pl_ser = pl.Series([1, 2, 3])
    pd_ser = cast("pd.Series[Any]", pd.Series([1, 2, 3]))
    pa_ser = cast("pa.ChunkedArray[Any]", pa.chunked_array([pa.array([1])]))  # type: ignore[redundant-cast]

    pl_1 = nw.from_native(pl_ser, series_only=True)
    pl_2 = nw.from_native(pl_ser, allow_series=True)
    pl_3 = nw.from_native(pl_ser, eager_only=True, series_only=True)
    pl_4 = nw.from_native(pl_ser, eager_only=True, series_only=True, allow_series=True)
    pl_5 = nw.from_native(pl_ser, eager_only=True, allow_series=True)
    pl_6 = nw.from_native(pl_ser, series_only=True, allow_series=True)
    pl_7 = nw.from_native(pl_ser, series_only=True, pass_through=True)
    pl_8 = nw.from_native(pl_ser, allow_series=True, pass_through=True)
    pl_9 = nw.from_native(pl_ser, eager_only=True, series_only=True, pass_through=True)
    pl_10 = nw.from_native(
        pl_ser, eager_only=True, series_only=True, allow_series=True, pass_through=True
    )
    pl_11 = nw.from_native(pl_ser, eager_only=True, allow_series=True, pass_through=True)
    pl_12 = nw.from_native(pl_ser, series_only=True, allow_series=True, pass_through=True)
    pls = pl_1, pl_2, pl_3, pl_4, pl_5, pl_6, pl_7, pl_8, pl_9, pl_10, pl_11, pl_12

    assert_type(pl_1, nw.Series[pl.Series])
    assert_type(pl_2, nw.Series[pl.Series])
    assert_type(pl_3, nw.Series[pl.Series])
    assert_type(pl_4, nw.Series[pl.Series])
    assert_type(pl_5, nw.Series[pl.Series])
    assert_type(pl_6, nw.Series[pl.Series])
    assert_type(pl_7, nw.Series[pl.Series])
    assert_type(pl_8, nw.Series[pl.Series])
    assert_type(pl_9, nw.Series[pl.Series])
    assert_type(pl_10, nw.Series[pl.Series])
    assert_type(pl_11, nw.Series[pl.Series])
    assert_type(pl_12, nw.Series[pl.Series])

    pd_1 = nw.from_native(pd_ser, series_only=True)
    pd_2 = nw.from_native(pd_ser, allow_series=True)
    pd_3 = nw.from_native(pd_ser, eager_only=True, series_only=True)
    pd_4 = nw.from_native(pd_ser, eager_only=True, series_only=True, allow_series=True)
    pd_5 = nw.from_native(pd_ser, eager_only=True, allow_series=True)
    pd_6 = nw.from_native(pd_ser, series_only=True, allow_series=True)
    pd_7 = nw.from_native(pd_ser, series_only=True, pass_through=True)
    pd_8 = nw.from_native(pd_ser, allow_series=True, pass_through=True)
    pd_9 = nw.from_native(pd_ser, eager_only=True, series_only=True, pass_through=True)
    pd_10 = nw.from_native(
        pd_ser, eager_only=True, series_only=True, allow_series=True, pass_through=True
    )
    pd_11 = nw.from_native(pd_ser, eager_only=True, allow_series=True, pass_through=True)
    pd_12 = nw.from_native(pd_ser, series_only=True, allow_series=True, pass_through=True)
    pds = pd_1, pd_2, pd_3, pd_4, pd_5, pd_6, pd_7, pd_8, pd_9, pd_10, pd_11, pd_12

    assert_type(pd_1, nw.Series["pd.Series[Any]"])
    assert_type(pd_2, nw.Series["pd.Series[Any]"])
    assert_type(pd_3, nw.Series["pd.Series[Any]"])
    assert_type(pd_4, nw.Series["pd.Series[Any]"])
    assert_type(pd_5, nw.Series["pd.Series[Any]"])
    assert_type(pd_6, nw.Series["pd.Series[Any]"])
    assert_type(pd_7, nw.Series["pd.Series[Any]"])
    assert_type(pd_8, nw.Series["pd.Series[Any]"])
    assert_type(pd_9, nw.Series["pd.Series[Any]"])
    assert_type(pd_10, nw.Series["pd.Series[Any]"])
    assert_type(pd_11, nw.Series["pd.Series[Any]"])
    assert_type(pd_12, nw.Series["pd.Series[Any]"])

    pa_1 = nw.from_native(pa_ser, series_only=True)
    pa_2 = nw.from_native(pa_ser, allow_series=True)
    pa_3 = nw.from_native(pa_ser, eager_only=True, series_only=True)
    pa_4 = nw.from_native(pa_ser, eager_only=True, series_only=True, allow_series=True)
    pa_5 = nw.from_native(pa_ser, eager_only=True, allow_series=True)
    pa_6 = nw.from_native(pa_ser, series_only=True, allow_series=True)
    pa_7 = nw.from_native(pa_ser, series_only=True, pass_through=True)
    pa_8 = nw.from_native(pa_ser, allow_series=True, pass_through=True)
    pa_9 = nw.from_native(pa_ser, eager_only=True, series_only=True, pass_through=True)
    pa_10 = nw.from_native(
        pa_ser, eager_only=True, series_only=True, allow_series=True, pass_through=True
    )
    pa_11 = nw.from_native(pa_ser, eager_only=True, allow_series=True, pass_through=True)
    pa_12 = nw.from_native(pa_ser, series_only=True, allow_series=True, pass_through=True)
    pas = pa_1, pa_2, pa_3, pa_4, pa_5, pa_6, pa_7, pa_8, pa_9, pa_10, pa_11, pa_12

    assert_type(pa_1, nw.Series["pa.ChunkedArray[Any]"])
    assert_type(pa_2, nw.Series["pa.ChunkedArray[Any]"])
    assert_type(pa_3, nw.Series["pa.ChunkedArray[Any]"])
    assert_type(pa_4, nw.Series["pa.ChunkedArray[Any]"])
    assert_type(pa_5, nw.Series["pa.ChunkedArray[Any]"])
    assert_type(pa_6, nw.Series["pa.ChunkedArray[Any]"])
    assert_type(pa_7, nw.Series["pa.ChunkedArray[Any]"])
    assert_type(pa_8, nw.Series["pa.ChunkedArray[Any]"])
    assert_type(pa_9, nw.Series["pa.ChunkedArray[Any]"])
    assert_type(pa_10, nw.Series["pa.ChunkedArray[Any]"])
    assert_type(pa_11, nw.Series["pa.ChunkedArray[Any]"])
    assert_type(pa_12, nw.Series["pa.ChunkedArray[Any]"])

    for series in chain(pls, pds, pas):
        assert isinstance(series, nw.Series)
