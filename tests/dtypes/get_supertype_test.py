from __future__ import annotations

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1  # noqa: F401
import narwhals.stable.v2 as nw_v2  # noqa: F401
from narwhals._utils import Version
from narwhals.dtypes import DType, get_supertype

XFAIL_TODO = pytest.mark.xfail(reason="TODO", raises=NotImplementedError)


@pytest.mark.parametrize(
    "dtype",
    [
        nw.Array(nw.Binary(), shape=(2,)),
        nw.Array(nw.Boolean, shape=(2, 3)),
        nw.Binary(),
        nw.Boolean(),
        nw.Categorical(),
        nw.Date(),
        nw.Datetime(),
        nw.Datetime(time_unit="ns", time_zone="Europe/Berlin"),
        nw.Decimal(),
        nw.Duration(),
        nw.Enum(["orca", "narwhal"]),
        nw.Float32(),
        nw.Float64(),
        nw.Int8(),
        nw.Int16(),
        nw.Int32(),
        nw.Int64(),
        nw.Int128(),
        nw.List(nw.String),
        nw.List(nw.Array(nw.Int32, shape=(5, 3))),
        nw.Object(),
        nw.String(),
        pytest.param(
            nw.Struct({"r2": nw.Float64(), "mse": nw.Float32()}), marks=XFAIL_TODO
        ),
        nw.Time(),
        nw.UInt8(),
        nw.UInt16(),
        nw.UInt32(),
        nw.UInt64(),
        nw.UInt128(),
        nw.Unknown(),
    ],
)
def test_same_dtype(dtype: DType) -> None:
    result = get_supertype(dtype, dtype, dtypes=Version.MAIN.dtypes)
    assert result is not None
    assert result == dtype


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        # signed + signed
        (nw.Int8(), nw.Int16(), nw.Int16()),
        (nw.Int8(), nw.Int32(), nw.Int32()),
        (nw.Int8(), nw.Int64(), nw.Int64()),
        (nw.Int8(), nw.Int128(), nw.Int128()),
        (nw.Int16(), nw.Int32(), nw.Int32()),
        (nw.Int16(), nw.Int64(), nw.Int64()),
        (nw.Int16(), nw.Int128(), nw.Int128()),
        (nw.Int32(), nw.Int64(), nw.Int64()),
        (nw.Int32(), nw.Int128(), nw.Int128()),
        (nw.Int64(), nw.Int128(), nw.Int128()),
        # unsigned + unsigned
        (nw.UInt8(), nw.UInt16(), nw.UInt16()),
        (nw.UInt8(), nw.UInt32(), nw.UInt32()),
        (nw.UInt8(), nw.UInt64(), nw.UInt64()),
        (nw.UInt8(), nw.UInt128(), nw.UInt128()),
        (nw.UInt16(), nw.UInt32(), nw.UInt32()),
        (nw.UInt16(), nw.UInt64(), nw.UInt64()),
        (nw.UInt16(), nw.UInt128(), nw.UInt128()),
        (nw.UInt32(), nw.UInt64(), nw.UInt64()),
        (nw.UInt32(), nw.UInt128(), nw.UInt128()),
        (nw.UInt64(), nw.UInt128(), nw.UInt128()),
        # signed + unsigned
        (nw.Int8(), nw.UInt8(), nw.Int16()),
        (nw.Int8(), nw.UInt16(), nw.Int32()),
        (nw.Int8(), nw.UInt32(), nw.Int64()),
        (nw.Int8(), nw.UInt64(), nw.Float64()),
        (nw.Int16(), nw.UInt8(), nw.Int16()),
        (nw.Int16(), nw.UInt16(), nw.Int32()),
        (nw.Int16(), nw.UInt32(), nw.Int64()),
        (nw.Int16(), nw.UInt64(), nw.Float64()),
        (nw.Int32(), nw.UInt8(), nw.Int32()),
        (nw.Int32(), nw.UInt16(), nw.Int32()),
        (nw.Int32(), nw.UInt32(), nw.Int64()),
        (nw.Int32(), nw.UInt64(), nw.Float64()),
        (nw.Int64(), nw.UInt8(), nw.Int64()),
        (nw.Int64(), nw.UInt16(), nw.Int64()),
        (nw.Int64(), nw.UInt32(), nw.Int64()),
        (nw.Int64(), nw.UInt64(), nw.Float64()),
        # float + float
        (nw.Float32(), nw.Float64(), nw.Float64()),
        # float + integer
        (nw.Int8(), nw.Float32(), nw.Float32()),
        (nw.Int16(), nw.Float32(), nw.Float32()),
        (nw.UInt8(), nw.Float32(), nw.Float32()),
        (nw.UInt16(), nw.Float32(), nw.Float32()),
        (nw.Int32(), nw.Float32(), nw.Float64()),
        (nw.Int64(), nw.Float32(), nw.Float64()),
        (nw.UInt32(), nw.Float32(), nw.Float64()),
        (nw.UInt64(), nw.Float32(), nw.Float64()),
        (nw.Int8(), nw.Float64(), nw.Float64()),
        (nw.Int64(), nw.Float64(), nw.Float64()),
        # numeric + boolean
        (nw.Int8(), nw.Boolean(), nw.Int8()),
        (nw.Int16(), nw.Boolean(), nw.Int16()),
        (nw.Int32(), nw.Boolean(), nw.Int32()),
        (nw.Int64(), nw.Boolean(), nw.Int64()),
        (nw.Int128(), nw.Boolean(), nw.Int128()),
        (nw.UInt8(), nw.Boolean(), nw.UInt8()),
        (nw.UInt16(), nw.Boolean(), nw.UInt16()),
        (nw.UInt32(), nw.Boolean(), nw.UInt32()),
        (nw.UInt64(), nw.Boolean(), nw.UInt64()),
        (nw.UInt128(), nw.Boolean(), nw.UInt128()),
        (nw.Decimal(), nw.Boolean(), nw.Decimal()),
        (nw.Float32(), nw.Boolean(), nw.Float32()),
        (nw.Float64(), nw.Boolean(), nw.Float64()),
    ],
)
def test_numeric_promotion(left: DType, right: DType, expected: DType) -> None:
    result = get_supertype(left, right, dtypes=Version.MAIN.dtypes)
    assert result is not None
    assert result == expected

    result = get_supertype(right, left, dtypes=Version.MAIN.dtypes)
    assert result is not None
    assert result == expected
