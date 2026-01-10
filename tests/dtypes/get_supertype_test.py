from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
import narwhals.stable.v2 as nw_v2  # noqa: F401
from narwhals._utils import Version
from narwhals.dtypes import DType
from narwhals.dtypes._supertyping import get_supertype

_Fn = TypeVar("_Fn", bound=Callable[..., Any])


def versions(fn: _Fn, /) -> _Fn:
    """Parametrize `version: Version`, but skip everything non-`MAIN` by default.

    Use this when a test case *should* have identical behavior between versions.
    """
    return pytest.mark.parametrize(
        "version",
        (
            Version.MAIN if v is Version.MAIN else pytest.param(v, marks=pytest.mark.slow)
            for v in Version
        ),
    )(fn)  # type: ignore[no-any-return]


XFAIL_DATE_NUMERIC = pytest.mark.xfail(reason="TODO: (Date, Numeric)")
XFAIL_TIME_NUMERIC = pytest.mark.xfail(reason="TODO: (Time, Numeric)")
XFAIL_DATETIME_NUMERIC = pytest.mark.xfail(reason="TODO: (Datetime, Numeric)")
XFAIL_DURATION_NUMERIC = pytest.mark.xfail(reason="TODO: (Duration, Numeric)")

XFAIL_V1 = pytest.mark.xfail(reason="TODO: V1 supertypes", raises=NotImplementedError)


def _dtype_ids(obj: DType | type[DType] | Version | None) -> str:  # noqa: PLR0911
    """Some tweaks to `DType.__repr__` for more readable test ids."""
    if obj is None:
        return str(obj)
    if isinstance(obj, Version):
        return obj.name
    if isinstance(obj, DType):
        if hasattr(obj, "__slots__"):
            if isinstance(obj, nw.Datetime):
                return f"Datetime[{obj.time_unit}, {obj.time_zone}]"
            if isinstance(obj, nw.Duration):
                return f"Duration[{obj.time_unit}]"
            if isinstance(obj, nw.Enum):
                return f"Enum{list(obj.categories)!r}"
            if isinstance(obj, nw.Array):
                dtype: Any = obj
                for _ in obj.shape:
                    dtype = dtype.inner
                return f"Array[{dtype!r}, {obj.shape}]"
            # non-empty slots == parameters
            return repr(obj)
        return obj.__class__.__name__
    return repr(obj)


@versions
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
        nw.Enum([]),
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
        nw.Struct({"r2": nw.Float64(), "mse": nw.Float32()}),
        nw.Struct({"a": nw.String, "b": nw.List(nw.Int32)}),
        nw.Time(),
        nw.UInt8(),
        nw.UInt16(),
        nw.UInt32(),
        nw.UInt64(),
        nw.UInt128(),
        nw.Unknown(),
    ],
    ids=_dtype_ids,
)
def test_identical_dtype(dtype: DType, version: Version) -> None:
    result = get_supertype(dtype, dtype, version)
    assert result is not None
    assert result == dtype


@versions
@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (nw.Datetime("ns"), nw.Datetime("us"), nw.Datetime("us")),
        (nw.Datetime("s"), nw.Datetime("us"), nw.Datetime("s")),
        (nw.Datetime("s"), nw.Datetime("s", "Africa/Accra"), None),
        (nw.Datetime(time_zone="Asia/Kathmandu"), nw.Datetime(), None),
        (
            nw.Enum(["beluga", "narwhal", "orca"]),
            nw.Enum(["dog", "cat", "fish with legs"]),
            None,
        ),
        (nw.Enum([]), nw.Enum(["fruit", "other food"]), None),
        (nw.List(nw.Int64), nw.List(nw.Int64()), nw.List(nw.Int64())),
        (nw.List(nw.UInt16()), nw.List(nw.Int32), nw.List(nw.Int32())),
        (nw.List(nw.Date), nw.List(nw.Binary), None),
        (nw.List(nw.Unknown), nw.List(nw.Float64), nw.List(nw.Unknown())),
        (
            nw.Array(nw.Float32, shape=2),
            nw.Array(nw.Float64, shape=2),
            nw.Array(nw.Float64, shape=2),
        ),
        (nw.Array(nw.Int64, shape=1), nw.Array(nw.Int64, shape=4), None),
        (
            nw.Array(nw.Decimal, shape=1),
            nw.Array(nw.Unknown, shape=1),
            nw.Array(nw.Unknown, shape=1),
        ),
        (
            nw.Array(nw.UInt128, shape=3),
            nw.Array(nw.Decimal, shape=3),
            nw.Array(nw.Decimal, shape=3),
        ),
        (nw.Array(nw.String, shape=1), nw.Array(nw.Int64, shape=1), None),
        (
            nw.Struct({"f0": nw.Duration("ms"), "f1": nw.Int64, "f2": nw.Int64}),
            nw.Struct({"f0": nw.Duration("us"), "f1": nw.Int64()}),
            nw.Struct({"f0": nw.Duration("ms"), "f1": nw.Int64(), "f2": nw.Int64()}),
        ),
        (
            nw.Struct({"f0": nw.Float64, "f1": nw.Date, "f2": nw.Int32}),
            nw.Struct({"f0": nw.Float32, "f1": nw.Datetime, "f3": nw.UInt8}),
            nw.Struct(
                {"f0": nw.Float64, "f1": nw.Datetime(), "f2": nw.Int32, "f3": nw.UInt8}
            ),
        ),
        (
            nw.Struct({"f0": nw.Int32, "f1": nw.Boolean, "f2": nw.String}),
            nw.Struct({"f0": nw.Unknown}),
            nw.Struct({"f0": nw.Unknown, "f1": nw.Boolean, "f2": nw.String}),
        ),
        (
            nw.Struct({"f0": nw.Object, "f1": nw.List(nw.Boolean)}),
            nw.Struct({"f0": nw.List(nw.Boolean), "f1": nw.List(nw.Boolean)}),
            None,
        ),
        (
            nw.Struct({"f0": nw.Binary()}),
            nw.Struct({"f0": nw.Datetime("s"), "f1": nw.Date}),
            None,
        ),
        (
            nw.Struct(
                {"f0": nw.Int64, "f1": nw.Struct({"f1": nw.Float32, "f0": nw.String})}
            ),
            nw.Struct(
                {
                    "f0": nw.UInt8,
                    "f1": nw.Struct(
                        {"f0": nw.Categorical, "f1": nw.Float64(), "f2": nw.Time}
                    ),
                }
            ),
            nw.Struct(
                {
                    "f0": nw.Int64,
                    "f1": nw.Struct({"f0": nw.String, "f1": nw.Float64, "f2": nw.Time}),
                }
            ),
        ),
        (
            nw.Struct({"F0": nw.UInt8, "f0": nw.Int16}),
            nw.Struct({"f0": nw.Int128, "f1": nw.UInt16, " f0": nw.Int8}),
            nw.Struct({"f0": nw.Int128, "f1": nw.UInt16, " f0": nw.Int8, "F0": nw.UInt8}),
        ),
    ],
    ids=_dtype_ids,
)
def test_same_class(
    left: DType, right: DType, expected: DType | None, version: Version
) -> None:
    result = get_supertype(left, right, version)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert result == expected


@versions
@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (nw.Datetime("ns"), nw.Date(), nw.Datetime("ns")),
        (nw.Date(), nw.Datetime(), nw.Datetime()),
        (nw.Datetime(), nw.Int8(), None),
        (nw.String(), nw.Categorical(), nw.String()),
        (nw.Enum(["hello"]), nw.Categorical(), None),
        (nw.Enum(["hello"]), nw.String(), nw.String()),
        (nw.Binary(), nw.String(), nw.Binary()),
        pytest.param(nw.Date(), nw.UInt32(), nw.Int32(), marks=XFAIL_DATE_NUMERIC),
        pytest.param(nw.Date(), nw.UInt64(), nw.Int64(), marks=XFAIL_DATE_NUMERIC),
        pytest.param(nw.Date(), nw.Int32(), nw.Int32(), marks=XFAIL_DATE_NUMERIC),
        pytest.param(nw.Date(), nw.Int64(), nw.Int64(), marks=XFAIL_DATE_NUMERIC),
        pytest.param(nw.Date(), nw.Float32(), nw.Float32(), marks=XFAIL_DATE_NUMERIC),
        pytest.param(nw.Date(), nw.Float64(), nw.Float64(), marks=XFAIL_DATE_NUMERIC),
        (nw.Time(), nw.UInt32(), None),
        (nw.Time(), nw.UInt64(), None),
        pytest.param(nw.Time(), nw.Int32(), nw.Int64(), marks=XFAIL_TIME_NUMERIC),
        pytest.param(nw.Time(), nw.Int64(), nw.Int64(), marks=XFAIL_TIME_NUMERIC),
        pytest.param(nw.Time(), nw.Float32(), nw.Float64(), marks=XFAIL_TIME_NUMERIC),
        pytest.param(nw.Time(), nw.Float64(), nw.Float64(), marks=XFAIL_TIME_NUMERIC),
        pytest.param(
            nw.Datetime(), nw.UInt32(), nw.Int64(), marks=XFAIL_DATETIME_NUMERIC
        ),
        pytest.param(
            nw.Datetime("s"), nw.UInt64(), nw.Int64(), marks=XFAIL_DATETIME_NUMERIC
        ),
        pytest.param(
            nw.Datetime("ns"), nw.Int32(), nw.Int64(), marks=XFAIL_DATETIME_NUMERIC
        ),
        pytest.param(nw.Datetime(), nw.Int64(), nw.Int64(), marks=XFAIL_DATETIME_NUMERIC),
        pytest.param(
            nw.Datetime("us"), nw.Float32(), nw.Float64(), marks=XFAIL_DATETIME_NUMERIC
        ),
        pytest.param(
            nw.Datetime("ms"), nw.Float64(), nw.Float64(), marks=XFAIL_DATETIME_NUMERIC
        ),
        pytest.param(
            nw.Duration(), nw.UInt32(), nw.Int64(), marks=XFAIL_DURATION_NUMERIC
        ),
        pytest.param(
            nw.Duration("s"), nw.UInt64(), nw.Int64(), marks=XFAIL_DURATION_NUMERIC
        ),
        pytest.param(
            nw.Duration("ns"), nw.Int32(), nw.Int64(), marks=XFAIL_DURATION_NUMERIC
        ),
        pytest.param(nw.Duration(), nw.Int64(), nw.Int64(), marks=XFAIL_DURATION_NUMERIC),
        pytest.param(
            nw.Duration("us"), nw.Float32(), nw.Float64(), marks=XFAIL_DURATION_NUMERIC
        ),
        pytest.param(
            nw.Duration("ms"), nw.Float64(), nw.Float64(), marks=XFAIL_DURATION_NUMERIC
        ),
    ],
    ids=_dtype_ids,
)
def test_mixed_dtype(
    left: DType, right: DType, expected: DType | None, version: Version
) -> None:
    result = get_supertype(left, right, version)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert result == expected


@versions
@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        # NOTE: The order of the case *should not* matter (some are flipped for coverage)
        # signed + signed
        (nw.Int8(), nw.Int16(), nw.Int16()),
        (nw.Int8(), nw.Int32(), nw.Int32()),
        (nw.Int8(), nw.Int64(), nw.Int64()),
        (nw.Int8(), nw.Int128(), nw.Int128()),
        (nw.Int16(), nw.Int32(), nw.Int32()),
        (nw.Int64(), nw.Int16(), nw.Int64()),
        (nw.Int16(), nw.Int128(), nw.Int128()),
        (nw.Int64(), nw.Int32(), nw.Int64()),
        (nw.Int32(), nw.Int128(), nw.Int128()),
        (nw.Int64(), nw.Int128(), nw.Int128()),
        # unsigned + unsigned
        (nw.UInt8(), nw.UInt16(), nw.UInt16()),
        (nw.UInt32(), nw.UInt8(), nw.UInt32()),
        (nw.UInt8(), nw.UInt64(), nw.UInt64()),
        (nw.UInt8(), nw.UInt128(), nw.UInt128()),
        (nw.UInt16(), nw.UInt32(), nw.UInt32()),
        (nw.UInt16(), nw.UInt64(), nw.UInt64()),
        (nw.UInt128(), nw.UInt16(), nw.UInt128()),
        (nw.UInt32(), nw.UInt64(), nw.UInt64()),
        (nw.UInt32(), nw.UInt128(), nw.UInt128()),
        (nw.UInt64(), nw.UInt128(), nw.UInt128()),
        # signed + unsigned
        (nw.Int8(), nw.UInt8(), nw.Int16()),
        (nw.Int8(), nw.UInt16(), nw.Int32()),
        (nw.UInt32(), nw.Int8(), nw.Int64()),
        (nw.Int8(), nw.UInt64(), nw.Float64()),
        (nw.Int16(), nw.UInt8(), nw.Int16()),
        (nw.Int16(), nw.UInt16(), nw.Int32()),
        (nw.UInt32(), nw.Int16(), nw.Int64()),
        (nw.Int16(), nw.UInt64(), nw.Float64()),
        (nw.Int32(), nw.UInt8(), nw.Int32()),
        (nw.UInt16(), nw.Int32(), nw.Int32()),
        (nw.Int32(), nw.UInt32(), nw.Int64()),
        (nw.Int32(), nw.UInt64(), nw.Float64()),
        (nw.Int64(), nw.UInt8(), nw.Int64()),
        (nw.UInt16(), nw.Int64(), nw.Int64()),
        (nw.Int64(), nw.UInt32(), nw.Int64()),
        (nw.Int64(), nw.UInt64(), nw.Float64()),
        # float + float
        (nw.Float32(), nw.Float64(), nw.Float64()),
        (nw.Float64(), nw.Float32(), nw.Float64()),
        # float + integer
        (nw.Int8(), nw.Float32(), nw.Float32()),
        (nw.Int16(), nw.Float32(), nw.Float32()),
        (nw.Float32(), nw.UInt8(), nw.Float32()),
        (nw.Float32(), nw.UInt16(), nw.Float32()),
        (nw.Int32(), nw.Float32(), nw.Float64()),
        (nw.Int64(), nw.Float32(), nw.Float64()),
        (nw.UInt32(), nw.Float32(), nw.Float64()),
        (nw.Float32(), nw.UInt64(), nw.Float64()),
        (nw.Int8(), nw.Float64(), nw.Float64()),
        (nw.Float64(), nw.Int64(), nw.Float64()),
        # numeric + boolean
        (nw.Int8(), nw.Boolean(), nw.Int8()),
        (nw.Int16(), nw.Boolean(), nw.Int16()),
        (nw.Int32(), nw.Boolean(), nw.Int32()),
        (nw.Boolean(), nw.Int64(), nw.Int64()),
        (nw.Int128(), nw.Boolean(), nw.Int128()),
        (nw.UInt8(), nw.Boolean(), nw.UInt8()),
        (nw.UInt16(), nw.Boolean(), nw.UInt16()),
        (nw.Boolean(), nw.UInt32(), nw.UInt32()),
        (nw.UInt64(), nw.Boolean(), nw.UInt64()),
        (nw.Boolean(), nw.UInt128(), nw.UInt128()),
        (nw.Decimal(), nw.Boolean(), nw.Decimal()),
        (nw.Boolean(), nw.Decimal(), nw.Decimal()),
        (nw.Float32(), nw.Boolean(), nw.Float32()),
        (nw.Boolean(), nw.Float64(), nw.Float64()),
    ],
    ids=_dtype_ids,
)
def test_numeric_promotion(
    left: DType, right: DType, expected: DType, version: Version
) -> None:
    result = get_supertype(left, right, version)
    assert result is not None
    assert result == expected

    result = get_supertype(left, right, version)
    assert result is not None
    assert result == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (nw_v1.Datetime(), nw_v1.Datetime(), nw_v1.Datetime()),
        (nw_v1.Datetime("ns"), nw_v1.Datetime("s"), nw_v1.Datetime("s")),
        (
            nw_v1.Datetime(time_zone="Europe/Berlin"),
            nw_v1.Datetime(time_zone="Europe/Berlin"),
            nw_v1.Datetime(time_zone="Europe/Berlin"),
        ),
        (
            nw_v1.Datetime(time_zone="Europe/Berlin"),
            nw_v1.Datetime("ms", "Europe/Berlin"),
            nw_v1.Datetime("ms", "Europe/Berlin"),
        ),
        (nw_v1.Datetime(time_zone="Europe/Berlin"), nw_v1.Datetime(), None),
        (nw_v1.Datetime("s"), nw_v1.Datetime("s", "Africa/Accra"), None),
        (nw_v1.Duration("ns"), nw_v1.Duration("ms"), nw_v1.Duration("ms")),
        (nw_v1.Duration(), nw_v1.Duration(), nw_v1.Duration()),
        (nw_v1.Duration("s"), nw_v1.Duration(), nw_v1.Duration("s")),
        (nw_v1.Duration(), nw_v1.Datetime(), None),
        (nw_v1.Enum(), nw_v1.Enum(), nw_v1.Enum()),
        (nw_v1.Enum(), nw_v1.String(), nw_v1.String()),
        (
            nw.Date(),
            nw_v1.Datetime(time_zone="Europe/Berlin"),
            nw_v1.Datetime(time_zone="Europe/Berlin"),
        ),
        (
            nw.Struct({"f0": nw_v1.Duration("ms"), "f1": nw.Int64, "f2": nw.Int64}),
            nw.Struct({"f0": nw_v1.Duration("us"), "f1": nw.Int64()}),
            nw.Struct({"f0": nw_v1.Duration("ms"), "f1": nw.Int64(), "f2": nw.Int64()}),
        ),
        (
            nw.Struct({"f0": nw.Float64, "f1": nw.Date, "f2": nw.Int32}),
            nw.Struct({"f0": nw.Float32, "f1": nw_v1.Datetime, "f3": nw.UInt8}),
            nw.Struct(
                {"f0": nw.Float64, "f1": nw_v1.Datetime(), "f2": nw.Int32, "f3": nw.UInt8}
            ),
        ),
        (
            nw.Struct({"f0": nw.Binary()}),
            nw.Struct({"f0": nw_v1.Datetime("s"), "f1": nw.Date}),
            None,
        ),
        (
            nw.Array(nw.Date, shape=3),
            nw.Array(nw_v1.Datetime, shape=3),
            nw.Array(nw_v1.Datetime, shape=3),
        ),
        (nw.Array(nw.Date, shape=1), nw.Array(nw_v1.Datetime, shape=3), None),
        (nw.Array(nw.Date, shape=2), nw.Array(nw_v1.Duration, shape=2), None),
        (
            nw.Array(nw_v1.Enum(), shape=4),
            nw.Array(nw_v1.Enum(), shape=4),
            nw.Array(nw_v1.Enum(), shape=4),
        ),
        # TODO @dangotbanned: What to do when e.g `(v1.Datetime, Datetime) -> ?`
    ],
)
def test_v1_dtypes(left: DType, right: DType, expected: DType | None) -> None:
    result = get_supertype(left, right, Version.V1)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert result == expected
        # Must also preserve v1-ness
        assert type(result) is type(expected)
