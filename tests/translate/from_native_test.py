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

# mypy: disallow-any-generics=false
from contextlib import nullcontext as does_not_raise
from importlib.util import find_spec
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, cast

import pytest

import narwhals as nw
from narwhals._utils import Version
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
    from sqlframe.duckdb import DuckDBSession

    session = DuckDBSession()
    df = session.createDataFrame([*zip(*data.values())], schema=[*data.keys()])

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
    from sqlframe.duckdb import DuckDBSession

    session = DuckDBSession()
    df = session.createDataFrame([*zip(*data.values())], schema=[*data.keys()])

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
        nw.from_native(mockdf)


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


@pytest.mark.parametrize(
    ("eager_only", "pass_through", "context"),
    [
        (False, False, does_not_raise()),
        (False, True, does_not_raise()),
        (True, True, does_not_raise()),
        (True, False, pytest.raises(TypeError, match="Cannot only use")),
    ],
)
def test_eager_only_pass_through_main(
    constructor: Constructor, *, eager_only: bool, pass_through: bool, context: Any
) -> None:
    if not any(s in str(constructor) for s in ("pyspark", "dask", "ibis", "duckdb")):
        pytest.skip(reason="Non lazy or polars")

    df = constructor(data)

    with context:
        res = nw.from_native(df, eager_only=eager_only, pass_through=pass_through)  # type: ignore[call-overload]
        if eager_only and pass_through:
            assert not isinstance(res, nw.LazyFrame)
        else:
            assert isinstance(res, nw.LazyFrame)
