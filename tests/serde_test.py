"""Serialization tests, based on [py-polars/tests/unit/test_serde.py].

See also [Pickling Class Instances](https://docs.python.org/3/library/pickle.html#pickling-class-instances).

[py-polars/tests/unit/test_serde.py]: https://github.com/pola-rs/polars/blob/a143eb0d7077ee9da2ce209a19c21d7f82228081/py-polars/tests/unit/test_serde.py
"""

from __future__ import annotations

import pickle

# ruff: noqa: S301
from typing import TYPE_CHECKING, Protocol, TypeVar

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.dtypes import DType
from narwhals.typing import IntoDType, TimeUnit

if TYPE_CHECKING:
    from narwhals.typing import DTypes


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
    assert result == namespace.Datetime(time_unit)


@namespaces
@time_units
def test_serde_duration_dtype(
    namespace: DTypes, time_unit: TimeUnit, roundtrip: Identity
) -> None:
    dtype = namespace.Duration(time_unit)
    result = roundtrip(dtype)
    assert result == namespace.Duration(time_unit)


def test_serde_categorical_dtype(roundtrip: Identity) -> None:
    dtype = nw.Categorical()
    result = roundtrip(dtype)
    assert result == nw.Categorical


def test_serde_doubly_nested_dtype(roundtrip: Identity) -> None:
    dtype = nw.Struct([nw.Field("a", nw.List(nw.String))])
    result = roundtrip(dtype)
    assert result == nw.Struct([nw.Field("a", nw.List(nw.String))])


def test_serde_array_dtype(roundtrip: Identity) -> None:
    dtype = nw.Array(nw.Int32(), 3)
    result = roundtrip(dtype)
    assert result == nw.Array(nw.Int32(), 3)


def test_serde_dtype_class(roundtrip: Identity) -> None:
    dtype_class = nw.Datetime
    result = roundtrip(dtype_class)
    assert result == dtype_class
    assert isinstance(result, type)


def test_serde_instantiated_dtype(roundtrip: Identity) -> None:
    dtype = nw.Int8()
    result = roundtrip(dtype)
    assert result == dtype
    assert isinstance(result, DType)


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
