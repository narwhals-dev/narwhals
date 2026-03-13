from __future__ import annotations

import decimal
import string

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals._dispatch import JustDispatch, just_dispatch
from narwhals.dtypes import DType, SignedIntegerType


def type_name(obj: object) -> str:
    return type(obj).__name__


@pytest.fixture
def dispatch_no_bound() -> JustDispatch[str]:
    @just_dispatch
    def dtype_repr_code(dtype: DType) -> str:
        return type_name(dtype).lower()

    return dtype_repr_code


@pytest.fixture
def dispatch_upper_bound() -> JustDispatch[str]:
    @just_dispatch(upper_bound=DType)
    def dtype_repr_code(dtype: DType) -> str:
        return type_name(dtype).lower()

    return dtype_repr_code


@pytest.fixture
def stdlib_decimal() -> decimal.Decimal:
    return decimal.Decimal("1.0")


def test_just_dispatch(
    dispatch_no_bound: JustDispatch[str], stdlib_decimal: decimal.Decimal
) -> None:
    i64 = nw.Int64()
    assert dispatch_no_bound(i64) == "int64"

    @dispatch_no_bound.register(*SignedIntegerType.__subclasses__())
    def repr_int(dtype: SignedIntegerType) -> str:
        return f"i{type_name(dtype).strip(string.ascii_letters)}"

    assert dispatch_no_bound(i64) == "i64"
    assert repr_int(i64) == "i64"
    assert dispatch_no_bound(nw.UInt8()) == "uint8"
    assert dispatch_no_bound(stdlib_decimal) == "decimal"


def test_just_dispatch_upper_bound(
    dispatch_upper_bound: JustDispatch[str], stdlib_decimal: decimal.Decimal
) -> None:
    i64 = nw.Int64()
    assert dispatch_upper_bound(i64) == "int64"

    @dispatch_upper_bound.register(*SignedIntegerType.__subclasses__())
    def repr_int(dtype: SignedIntegerType) -> str:
        return f"i{type_name(dtype).strip(string.ascii_letters)}"

    assert dispatch_upper_bound(i64) == "i64"
    assert repr_int(i64) == "i64"
    assert dispatch_upper_bound(nw.UInt8()) == "uint8"
    assert dispatch_upper_bound(nw.Decimal()) == "decimal"

    with pytest.raises(TypeError, match=r"'dtype_repr_code' does not support 'Decimal'"):
        dispatch_upper_bound(stdlib_decimal)

    dispatch_upper_bound.register(type(stdlib_decimal))(lambda _: "need to be explicit")
    assert dispatch_upper_bound(stdlib_decimal) == "need to be explicit"


def test_just_dispatch_no_auto_subclass(dispatch_no_bound: JustDispatch[str]) -> None:
    NOT_REGISTERED = "datetime"  # noqa: N806
    TZ = "Africa/Accra"  # noqa: N806

    assert dispatch_no_bound(nw.Datetime("ms")) == NOT_REGISTERED
    assert dispatch_no_bound(nw_v1.Datetime("us")) == NOT_REGISTERED

    @dispatch_no_bound.register(nw.Datetime)
    def repr_datetime(dtype: nw.Datetime) -> str:
        if dtype.time_zone is None:
            args: str = dtype.time_unit
        else:
            args = f"{dtype.time_unit}, {dtype.time_zone}"
        return f"datetime[{args}]"

    assert dispatch_no_bound(nw.Datetime()) == "datetime[us]"
    assert dispatch_no_bound(nw.Datetime("s")) == "datetime[s]"
    assert dispatch_no_bound(nw.Datetime(time_zone=TZ)) == f"datetime[us, {TZ}]"

    assert dispatch_no_bound(nw_v1.Datetime()) == NOT_REGISTERED
    assert dispatch_no_bound(nw_v1.Datetime("s")) == NOT_REGISTERED
    assert dispatch_no_bound(nw_v1.Datetime(time_zone=TZ)) == NOT_REGISTERED

    dispatch_no_bound.register(nw_v1.Datetime)(repr_datetime)

    assert dispatch_no_bound(nw_v1.Datetime()) == "datetime[us]"
    assert dispatch_no_bound(nw_v1.Datetime("s")) == "datetime[s]"
    assert dispatch_no_bound(nw_v1.Datetime(time_zone=TZ)) == f"datetime[us, {TZ}]"
