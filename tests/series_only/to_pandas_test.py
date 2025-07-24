from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import narwhals as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="too old for pyarrow")
def test_convert(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    data = [1, 3, 2]
    if any(
        cname in str(constructor_eager)
        for cname in ("pandas_nullable", "pandas_pyarrow", "modin_pyarrow")
    ):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].alias(
        "a"
    )

    result = series.to_pandas()
    assert_series_equal(result, pd.Series([1, 3, 2], name="a"))


@pytest.mark.skipif(PANDAS_VERSION < (1, 5, 0), reason="too old for pyarrow")
@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            {"a": [1, 3, 8], "b": [4.1, 2.3, 3.0]},
            nw.Struct({"a": nw.Int64, "b": nw.Float64}),
        ),
        (
            {"a": [dt.datetime(2000, 1, 1), dt.datetime(2000, 1, 2)], "b": ["one", None]},
            nw.Struct({"a": nw.Datetime(), "b": nw.String}),
        ),
    ],
)
def test_pyarrow_to_pandas_struct(data: dict[str, Any], expected: nw.Struct) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    struct_array = pa.table(data).to_struct_array()
    struct_series_pa = nw.from_native(struct_array, series_only=True).alias("c")
    struct_series_pd = nw.from_native(struct_series_pa.to_pandas(), series_only=True)

    assert struct_series_pd.dtype == expected
    assert struct_series_pd.dtype == struct_series_pa.dtype
    assert struct_series_pd.name == struct_series_pa.name
