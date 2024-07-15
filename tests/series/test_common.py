from __future__ import annotations

import re
from typing import Any
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

import narwhals.stable.v1 as nw
from narwhals._pandas_like.utils import Implementation
from narwhals.dependencies import get_dask
from narwhals.utils import parse_version
from tests.conftest import dask_series_constructor

data = [1, 3, 2]
data_dups = [4, 4, 6]
data_sorted = [7.0, 8, 9]


def compute_if_dask(result: Any) -> Any:
    if (
        hasattr(result, "_native_series")
        and hasattr(result._native_series, "_implementation")
        and result._series._implementation is Implementation.DASK
    ):
        return result.to_pandas()
    return result


def test_len(constructor_series: Any) -> None:
    series = nw.from_native(constructor_series(data), series_only=True)

    result = len(series)
    assert result == 3

    result = series.len()
    assert result == 3


def test_is_in(request: Any, constructor_series: Any) -> None:
    if "pyarrow_series" in str(constructor_series):
        request.applymarker(pytest.mark.xfail)
    series = nw.from_native(constructor_series(data), series_only=True)

    result = series.is_in([1, 2])
    assert result[0]
    assert not result[1]
    assert result[2]


def test_is_in_other(constructor: Any) -> None:
    df_raw = constructor({"a": data})
    with pytest.raises(
        NotImplementedError,
        match=(
            "Narwhals `is_in` doesn't accept expressions as an argument, as opposed to Polars. You should provide an iterable instead."
        ),
    ):
        nw.from_native(df_raw).with_columns(contains=nw.col("a").is_in("sets"))


def test_dtype(constructor_series: Any) -> None:
    series = nw.from_native(constructor_series(data), series_only=True)
    result = series.dtype
    assert result == nw.Int64
    assert result.is_numeric()


def test_reductions(request: Any, constructor_series: Any) -> None:
    if "pyarrow_series" in str(constructor_series):
        request.applymarker(pytest.mark.xfail)

    s = nw.from_native(constructor_series(data), series_only=True)
    assert s.mean() == 2.0
    assert s.std() == 1.0
    assert s.min() == 1
    assert s.max() == 3
    assert s.count() == 3
    assert s.sum() == 6
    assert nw.to_native(s.is_between(1, 2))[0]
    assert not nw.to_native(s.is_between(1, 2))[1]
    assert nw.to_native(s.is_between(1, 2))[2]
    assert s.n_unique() == 3
    unique = s.unique().sort()
    assert unique[0] == 1
    assert unique[1] == 2
    assert unique[2] == 3
    assert s.alias("foo").name == "foo"


def test_boolean_reductions(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df_raw = constructor({"a": data})
    df = nw.from_native(df_raw).lazy().select(nw.col("a") > 1)
    assert not df.collect()["a"].all()
    assert df.collect()["a"].any()


@pytest.mark.skipif(
    parse_version(pd.__version__) < parse_version("2.0.0"), reason="too old for pyarrow"
)
def test_convert(request: Any, constructor_series: Any) -> None:
    if any(
        cname in str(constructor_series)
        for cname in ("pyarrow_series", "pandas_series_nullable", "pandas_series_pyarrow")
    ):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_series(data).rename("a"), series_only=True)

    result = series.to_numpy()
    assert_array_equal(result, np.array([1, 3, 2]))

    result = series.to_pandas()
    assert_series_equal(result, pd.Series([1, 3, 2], name="a"))


def test_to_numpy() -> None:
    s = pd.Series([1, 2, None], dtype="Int64")
    nw_series = nw.from_native(s, series_only=True)
    assert nw_series.to_numpy().dtype == "float64"
    assert nw_series.__array__().dtype == "float64"
    assert nw_series.shape == (3,)


def test_is_duplicated(request: Any, constructor_series: Any) -> None:
    if "pyarrow_series" in str(constructor_series):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_series(data_dups), series_only=True)
    result = series.is_duplicated()
    result = compute_if_dask(result)
    expected = np.array([True, True, False])
    assert (result.to_numpy() == expected).all()


def test_is_unique(request: Any, constructor_series: Any) -> None:
    if "pyarrow_series" in str(constructor_series):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_series(data_dups), series_only=True)
    result = series.is_unique()
    expected = np.array([False, False, True])
    assert (result.to_numpy() == expected).all()


def test_is_first_distinct(request: Any, constructor_series: Any) -> None:
    if "pyarrow_series" in str(constructor_series):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_series(data_dups), series_only=True)
    result = series.is_first_distinct()
    expected = np.array([True, False, True])
    assert (result.to_numpy() == expected).all()


def test_is_last_distinct(request: Any, constructor_series: Any) -> None:
    if "pyarrow_series" in str(constructor_series):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_series(data_dups), series_only=True)
    result = series.is_last_distinct()
    expected = np.array([False, True, True])
    assert (result.to_numpy() == expected).all()


def test_value_counts(request: Any, constructor_series: Any) -> None:
    if "pyarrow_series" in str(constructor_series):
        request.applymarker(pytest.mark.xfail)

    if "pandas_series_nullable" in str(constructor_series):  # fails for py3.8
        pytest.skip()

    series = nw.from_native(constructor_series(data_dups).rename("b"), series_only=True)

    sorted_result = series.value_counts(sort=True)
    assert sorted_result.columns == ["b", "count"]

    expected = np.array([[4, 2], [6, 1]])
    assert (sorted_result.to_numpy() == expected).all()

    unsorted_result = series.value_counts(sort=False)
    assert unsorted_result.columns == ["b", "count"]

    a = unsorted_result.to_numpy()

    assert (a[a[:, 0].argsort()] == expected).all()


@pytest.mark.parametrize(
    ("input_data", "descending", "expected"),
    [(data, False, False), (data_sorted, False, True), (data_sorted, True, False)],
)
def test_is_sorted(
    request: Any,
    constructor_series: Any,
    input_data: str,
    descending: bool,  # noqa: FBT001
    expected: bool,  # noqa: FBT001
) -> None:
    if "pyarrow_series" in str(constructor_series):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_series(input_data), series_only=True)
    result = series.is_sorted(descending=descending)
    if (dd := get_dask()) is not None and isinstance(df_raw, dd.DataFrame):
        result = result.compute()
    assert result == expected


def test_is_sorted_invalid(request: Any, constructor_series: Any) -> None:
    if "pyarrow_series" in str(constructor_series):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_series(data_sorted), series_only=True)

    with pytest.raises(TypeError):
        series.is_sorted(descending="invalid_type")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("interpolation", "expected"),
    [
        ("lower", 7.0),
        ("higher", 8.0),
        ("midpoint", 7.5),
        ("linear", 7.6),
        ("nearest", 8.0),
    ],
)
@pytest.mark.filterwarnings("ignore:the `interpolation=` argument to percentile")
def test_quantile(
    request: Any,
    constructor_series: Any,
    interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    expected: float,
) -> None:
    if "pyarrow_series" in str(constructor_series):
        request.applymarker(pytest.mark.xfail)

    q = 0.3
    if is_dask_test := constructor_series == dask_series_constructor:
        interpolation = "linear"  # other interpolation unsupported in dask

    series = nw.from_native(constructor_series(data_sorted), allow_series=True)

    result = series.quantile(quantile=q, interpolation=interpolation)  # type: ignore[union-attr]
    if is_dask_test:
        result = result.compute()
    assert result == expected


def test_zip_with(request: Any, constructor_series: Any) -> None:
    if "pyarrow_series" in str(constructor_series):
        request.applymarker(pytest.mark.xfail)

    series1 = nw.from_native(constructor_series(data), series_only=True)
    series2 = nw.from_native(constructor_series(data_dups), series_only=True)
    mask = nw.from_native(constructor_series([True, False, True]), series_only=True)

    result = series1.zip_with(mask, series2)
    expected = [1, 4, 2]
    assert result.to_list() == expected


@pytest.mark.skipif(
    parse_version(pd.__version__) < parse_version("1.0.0"),
    reason="too old for convert_dtypes",
)
def test_cast_string() -> None:
    s_pd = pd.Series([1, 2]).convert_dtypes()
    s = nw.from_native(s_pd, series_only=True)
    s = s.cast(nw.String)
    result = nw.to_native(s)
    assert str(result.dtype) in ("string", "object", "dtype('O')")


@pytest.mark.parametrize(("index", "expected"), [(0, 1), (1, 3)])
def test_item(constructor_series: Any, index: int, expected: int) -> None:
    series = nw.from_native(constructor_series(data), series_only=True)
    result = series.item(index)
    assert result == expected
    assert series.head(1).item() == 1

    with pytest.raises(
        ValueError,
        match=re.escape("can only call '.item()' if the Series is of length 1,"),
    ):
        series.item(None)
