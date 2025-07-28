from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import narwhals as nw
from narwhals._utils import Version
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
def test_pyarrow_to_pandas_use_pyarrow(data: dict[str, Any], expected: nw.Struct) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    arrow_namespace = Version.MAIN.namespace.from_backend("pyarrow").compliant
    ser_pa = arrow_namespace.from_native(pa.table(data)).to_struct("c").to_narwhals()
    ser_pd = nw.from_native(
        ser_pa.to_pandas(use_pyarrow_extension_array=True), series_only=True
    )
    assert ser_pd.dtype == expected
    assert ser_pd.dtype == ser_pa.dtype
    assert ser_pd.name == ser_pa.name
