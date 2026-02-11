"""Serialization tests, based on [py-polars/tests/unit/test_serde.py].

See also [Pickling Class Instances](https://docs.python.org/3/library/pickle.html#pickling-class-instances).

[py-polars/tests/unit/test_serde.py]: https://github.com/pola-rs/polars/blob/a143eb0d7077ee9da2ce209a19c21d7f82228081/py-polars/tests/unit/test_serde.py
"""

from __future__ import annotations

import pickle
import string

# ruff: noqa: S301
from typing import TYPE_CHECKING, Protocol, TypeVar

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.dtypes import DType
from narwhals.typing import IntoDType, NonNestedDType, TimeUnit

if TYPE_CHECKING:
    from narwhals.typing import DTypes
    from tests.utils import NestedOrEnumDType


IntoDTypeT = TypeVar("IntoDTypeT", bound=IntoDType)


namespaces = pytest.mark.parametrize("namespace", [nw, nw_v1])
time_units = pytest.mark.parametrize("time_unit", ["ns", "us", "ms", "s"])


class Identity(Protocol):
    def __call__(self, obj: IntoDTypeT, /) -> IntoDTypeT: ...


def _roundtrip_pickle(protocol: int | None = None) -> Identity:
    def fn(obj: IntoDTypeT, /) -> IntoDTypeT:
        result: IntoDTypeT = pickle.loads(pickle.dumps(obj, protocol))
        return result

    return fn


@pytest.fixture(
    params=[_roundtrip_pickle(), _roundtrip_pickle(4), _roundtrip_pickle(5)],
    ids=["pickle-None", "pickle-4", "pickle-5"],
)
def roundtrip(request: pytest.FixtureRequest) -> Identity:
    fn: Identity = request.param
    return fn


@namespaces
@time_units
def test_serde_datetime_dtype(
    namespace: DTypes, time_unit: TimeUnit, roundtrip: Identity
) -> None:
    dtype = namespace.Datetime(time_unit)
    result = roundtrip(dtype)
    assert result == dtype


@namespaces
@time_units
def test_serde_duration_dtype(
    namespace: DTypes, time_unit: TimeUnit, roundtrip: Identity
) -> None:
    dtype = namespace.Duration(time_unit)
    result = roundtrip(dtype)
    assert result == dtype


def test_serde_doubly_nested_struct_dtype(roundtrip: Identity) -> None:
    dtype = nw.Struct([nw.Field("a", nw.List(nw.String))])
    result = roundtrip(dtype)
    assert result == dtype


def test_serde_doubly_nested_array_dtype(roundtrip: Identity) -> None:
    dtype = nw.Array(nw.Array(nw.Int32(), 2), 3)
    result = roundtrip(dtype)
    assert result == dtype


def test_serde_dtype_class(roundtrip: Identity) -> None:
    dtype_class = nw.Datetime
    result = roundtrip(dtype_class)
    assert result == dtype_class
    assert isinstance(result, type)


def test_serde_enum_dtype(roundtrip: Identity) -> None:
    dtype = nw.Enum(["a", "b"])
    result = roundtrip(dtype)
    assert result == dtype
    assert isinstance(result, DType)


def test_serde_enum_v1_dtype(roundtrip: Identity) -> None:
    dtype = nw_v1.Enum()
    result = roundtrip(dtype)
    assert result == dtype
    assert isinstance(result, nw_v1.Enum)
    tp = type(result)
    with pytest.raises(TypeError):
        tp(["a", "b"])  # type: ignore[call-arg]


def test_serde_enum_deferred(roundtrip: Identity) -> None:
    pytest.importorskip("polars")
    import polars as pl

    categories = tuple(string.printable)
    dtype_pl = pl.Enum(categories)
    series_pl = pl.Series(categories).cast(dtype_pl).extend_constant(categories[5], 1000)
    series_nw = nw.from_native(series_pl, series_only=True)
    dtype_nw = series_nw.dtype
    assert isinstance(dtype_nw, nw.Enum)
    result = roundtrip(dtype_nw)
    assert isinstance(result, nw.Enum)
    assert result == dtype_nw
    assert result == series_nw.dtype
    assert dtype_nw == roundtrip(result)
    assert (
        type(result)(dtype_pl.categories).categories
        == roundtrip(result).categories
        == categories
        == result.categories
        == roundtrip(dtype_nw).categories
    )


def test_serde_non_nested_dtypes(
    non_nested_type: type[NonNestedDType], roundtrip: Identity
) -> None:
    dtype = non_nested_type()
    result = roundtrip(dtype)
    assert isinstance(result, DType)
    assert isinstance(result, non_nested_type)
    assert result == non_nested_type()
    assert result == non_nested_type


def test_serde_nested_dtypes(
    nested_dtype: NestedOrEnumDType, roundtrip: Identity
) -> None:
    result = roundtrip(nested_dtype)
    assert isinstance(result, DType)
    assert isinstance(result, nested_dtype.__class__)
    assert result == nested_dtype
    assert result == nested_dtype.base_type()
