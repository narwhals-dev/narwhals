from __future__ import annotations

import datetime as dt
from copy import copy, deepcopy
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals._plan.schema import FrozenSchema

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.plan.utils import DataFrame


def test_schema() -> None:
    mapping = {"a": nw.Int64(), "b": nw.String()}
    schema = nw.Schema(mapping)
    frozen_schema = FrozenSchema(mapping)

    assert frozen_schema.keys() == schema.keys()
    assert tuple(frozen_schema.values()) == tuple(schema.values())

    # NOTE: Would type-check if `Schema.__init__` didn't make liskov unhappy
    assert schema == nw.Schema(frozen_schema)  # type: ignore[arg-type]
    assert mapping == dict(frozen_schema)

    # NOTE: This is an internal version of `Schema`, but is not interchangeable
    assert frozen_schema != mapping
    assert frozen_schema != schema

    assert frozen_schema == FrozenSchema(mapping)
    assert frozen_schema == FrozenSchema(**mapping)
    assert frozen_schema == FrozenSchema(a=nw.Int64(), b=nw.String())
    assert frozen_schema == FrozenSchema(schema)
    assert frozen_schema == FrozenSchema(frozen_schema)
    assert frozen_schema == FrozenSchema(frozen_schema.items())

    # NOTE: Using `**` unpacking, despite not inheriting from `Mapping` or `dict`
    assert frozen_schema == FrozenSchema(**frozen_schema)

    # NOTE: In case this all looks *too good* to be true
    assert frozen_schema != FrozenSchema(**mapping, c=nw.Float64())

    assert frozen_schema["a"] == schema["a"]

    assert frozen_schema.get("c") is None
    assert frozen_schema.get("c", nw.Unknown) is nw.Unknown
    assert frozen_schema.get("c", nw.Unknown()) == nw.Unknown()

    assert "b" in frozen_schema
    assert "e" not in frozen_schema

    with pytest.raises(TypeError, match="Cannot subclass 'FrozenSchema'"):

        class MutableSchema(FrozenSchema): ...  # type: ignore[misc]


def test_from_has_schema(dataframe: DataFrame) -> None:
    data = {"a": [{"a": dt.datetime(2001, 1, 1)}], "b": [dt.time(1, 1, 1)]}
    expected = FrozenSchema({"a": nw.Struct({"a": nw.Datetime()}), "b": nw.Time()})
    df = dataframe(data)
    assert expected == FrozenSchema(df)
    assert expected is FrozenSchema(df)
    assert expected is FrozenSchema(dataframe(deepcopy(data)))


def test_schema_hash() -> None:
    mapping = {
        "a": nw.List(nw.Float32()),
        "b": nw.Struct({"a": nw.String(), "c": nw.Boolean()}),
    }
    schema_1 = FrozenSchema(mapping)
    schema_2 = FrozenSchema(**mapping)

    hash_1 = hash(schema_1)
    hash_2 = hash(schema_2)
    assert hash_1 == hash_2


@pytest.mark.parametrize("function", [copy, deepcopy], ids=["copy", "deepcopy"])
def test_schema_copy(function: Callable[[FrozenSchema], FrozenSchema]) -> None:
    # See https://github.com/narwhals-dev/narwhals/blob/41d8c8e06240b8cdfbfb85082ce8a73bdc5fae12/tests/plan/immutable_test.py#L129-L136
    schema = FrozenSchema({"a": nw.Date(), "b": nw.String(), "c": nw.Binary()})
    clone = function(schema)
    assert clone == schema
    assert clone is schema


def test_schema_setattr_delattr__() -> None:
    mapping = {"a": nw.Date(), "b": nw.String(), "c": nw.Binary()}
    schema = FrozenSchema(mapping)
    with pytest.raises(AttributeError, match=r"FrozenSchema.+immutable.+'_mapping'"):
        schema._mapping = mapping  # type: ignore[assignment]
    with pytest.raises(
        AttributeError, match=r"'FrozenSchema' object has no attribute 'hey_i_dont_exist'"
    ):
        schema.hey_i_dont_exist = 1  # type: ignore[assignment]
    with pytest.raises(AttributeError, match=r"FrozenSchema.+immutable.+'_mapping'"):
        del schema._mapping
    with pytest.raises(
        AttributeError, match=r"'FrozenSchema' object has no attribute 'me_either'"
    ):
        del schema.me_either  # type: ignore[attr-defined]
