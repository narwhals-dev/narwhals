from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import compare_dicts

data = [1, 3, 2]
data_dups = [4, 4, 6]
data_sorted = [7.0, 8, 9]


def test_len(constructor_eager: Any) -> None:
    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]

    result = len(series)
    assert result == 3

    result = series.len()
    assert result == 3


def test_is_in(constructor_eager: Any) -> None:
    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]

    result = series.is_in([1, 2]).to_list()
    assert result[0]
    assert not result[1]
    assert result[2]


def test_is_in_other(constructor_eager: Any) -> None:
    df_raw = constructor_eager({"a": data})
    with pytest.raises(
        NotImplementedError,
        match=(
            "Narwhals `is_in` doesn't accept expressions as an argument, as opposed to Polars. You should provide an iterable instead."
        ),
    ):
        nw.from_native(df_raw).with_columns(contains=nw.col("a").is_in("sets"))


def test_dtype(constructor_eager: Any) -> None:
    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    result = series.dtype
    assert result == nw.Int64
    assert result.is_numeric()


def test_reductions(request: Any, constructor_eager: Any) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
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


def test_boolean_reductions(request: Any, constructor_eager: Any) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df_raw = constructor_eager({"a": data})
    df = nw.from_native(df_raw).lazy().select(nw.col("a") > 1)
    assert not df.collect()["a"].all()
    assert df.collect()["a"].any()


@pytest.mark.skipif(
    parse_version(pd.__version__) < parse_version("2.0.0"), reason="too old for pyarrow"
)
def test_convert(request: Any, constructor_eager: Any) -> None:
    if any(
        cname in str(constructor_eager)
        for cname in ("pandas_nullable", "pandas_pyarrow", "modin")
    ):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].alias(
        "a"
    )

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


def test_zip_with(constructor_eager: Any) -> None:
    series1 = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    series2 = nw.from_native(constructor_eager({"a": data_dups}), eager_only=True)["a"]
    mask = nw.from_native(constructor_eager({"a": [True, False, True]}), eager_only=True)[
        "a"
    ]

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
def test_item(constructor_eager: Any, index: int, expected: int) -> None:
    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    result = series.item(index)
    compare_dicts({"a": [result]}, {"a": [expected]})
    compare_dicts({"a": [series.head(1).item()]}, {"a": [1]})

    with pytest.raises(
        ValueError,
        match=re.escape("can only call '.item()' if the Series is of length 1,"),
    ):
        series.item(None)
