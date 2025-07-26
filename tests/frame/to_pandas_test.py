from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


@pytest.mark.filterwarnings("ignore:.*Passing a BlockManager.*:DeprecationWarning")
@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="too old for pandas-pyarrow")
def test_convert_pandas(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = constructor_eager(data)
    result = nw.from_native(df_raw, eager_only=True).to_pandas()

    if constructor_eager.__name__.startswith("pandas"):
        expected = cast("pd.DataFrame", constructor_eager(data))
    elif "modin_pyarrow" in str(constructor_eager):
        expected = pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow")
    else:
        expected = pd.DataFrame(data)

    pd.testing.assert_frame_equal(result, expected)


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

    expected_schema = nw.Schema({"c": expected})

    struct_array = pa.table(data).to_struct_array()
    struct_frame_pa = nw.from_native(struct_array, series_only=True).alias("c").to_frame()
    struct_frame_pd = nw.from_native(struct_frame_pa.to_pandas())

    assert struct_frame_pd.schema == expected_schema
    assert struct_frame_pd.schema == struct_frame_pa.schema
