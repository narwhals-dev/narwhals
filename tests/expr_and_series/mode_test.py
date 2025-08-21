from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import narwhals as nw
from narwhals.exceptions import ShapeError
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 1, 2, 2, 3], "b": [1, 2, 3, 3, 4]}
data_group = {
    "grp": ["g1", "g1", "g1", "g1", "g2", "g2", "g2"],
    "vals_unimodal": [1, 1, 2, 3, 3, 3, 4],
    "vals_multimodal": [1, 1, 2, 2, 3, 3, 4],
}


def test_mode_single_expr_keep_all(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").mode(keep="all")).sort("a")
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)


def test_mode_series_keep_all(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.mode(keep="all").sort()
    expected = {"a": [1, 2]}
    assert_equal_data({"a": result}, expected)


def test_mode_different_lengths_keep_all(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 10):
        pytest.skip()
    df = nw.from_native(constructor_eager(data))
    with pytest.raises(ShapeError):
        df.select(nw.col("a", "b").mode(keep="all"))


def test_mode_expr_keep_any(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b").mode(keep="any")).lazy().collect()

    try:
        expected = {"a": [1], "b": [3]}
        assert_equal_data(result, expected)
    except AssertionError:  # pragma: no cover
        expected = {"a": [2], "b": [3]}
        assert_equal_data(result, expected)


def test_mode_expr_keep_all_lazy(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    impl = df.implementation
    not_implemented = {
        nw.Implementation.DUCKDB,
        nw.Implementation.IBIS,
        nw.Implementation.PYSPARK,
        nw.Implementation.PYSPARK_CONNECT,
        nw.Implementation.SQLFRAME,
    }
    msg = "`keep='all'` is not implemented for backend"
    context = (
        pytest.raises(NotImplementedError, match=msg)
        if impl in not_implemented
        else does_not_raise()
    )

    with context:
        result = df.select(nw.col("a").mode(keep="all").sum())
        assert_equal_data(result, {"a": [3]})


def test_mode_group_by_unimodal(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    df = nw.from_native(constructor(data_group))
    impl = df.implementation

    if impl.is_pyarrow() or impl.is_dask() or impl.is_modin():
        # Tracker:
        #   - Pyarrow: https://github.com/apache/arrow/issues/20359
        #   - Dask: TODO(FBruzzesi)
        #   - Modin: TODO(FBruzzesi) - Currently raises the following exception:
        #       >  TypeError: super(type, obj): obj must be an instance or subtype of type
        request.applymarker(pytest.mark.xfail)

    result = (
        df.group_by("grp")
        .agg(nw.col("vals_unimodal").mode(keep="any"))
        .sort("grp")
        .lazy()
        .collect()
    )
    expected = {"grp": ["g1", "g2"], "vals_unimodal": [1, 3]}
    assert_equal_data(result, expected)


def test_mode_group_by_multimodal(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    df = nw.from_native(constructor(data_group))
    impl = df.implementation

    if impl.is_pandas_like() or impl.is_pyarrow() or impl.is_dask():
        # Tracker:
        #   - Pandas: Multimodal is not supported
        #   - Pyarrow: https://github.com/apache/arrow/issues/20359
        #   - Dask: TODO(FBruzzesi)

        request.applymarker(pytest.mark.xfail)

    result = (
        df.group_by("grp")
        .agg(nw.col("vals_multimodal").mode(keep="any"))
        .sort("grp")
        .lazy()
        .collect()
    )
    try:
        expected = {"grp": ["g1", "g2"], "vals_multimodal": [1, 3]}
        assert_equal_data(result, expected)
    except AssertionError:  # pragma: no cover
        expected = {"grp": ["g1", "g2"], "vals_multimodal": [2, 3]}
        assert_equal_data(result, expected)
