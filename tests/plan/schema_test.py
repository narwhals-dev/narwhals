from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from copy import copy, deepcopy
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals._plan.schema import FrozenSchema, IntoSchema
from narwhals._typing_compat import assert_never

if TYPE_CHECKING:
    from collections.abc import Callable

    from narwhals.dtypes import DType
    from tests.plan.utils import DataFrame

F64, F32, I64, I32, STR = nw.Float64(), nw.Float32(), nw.Int64(), nw.Int32(), nw.String()
BINARY, BOOL, DATE, TIME = nw.Binary(), nw.Boolean(), nw.Date(), nw.Time()


# TODO @dangotbanned: Split this up into at least 3 tests
def test_schema() -> None:
    mapping = {"a": I64, "b": STR}
    schema = nw.Schema(mapping)
    frozen_schema = FrozenSchema(mapping)

    assert frozen_schema.keys() == schema.keys()
    assert tuple(frozen_schema.values()) == tuple(schema.values())

    assert schema == nw.Schema(frozen_schema)
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

    # Make sure typing check works first
    if not isinstance(frozen_schema, Mapping):
        assert_never(frozen_schema)
    assert isinstance(frozen_schema, Mapping)
    assert Mapping not in FrozenSchema.__bases__


def test_from_has_schema(dataframe: DataFrame) -> None:
    data = {"a": [{"a": dt.datetime(2001, 1, 1)}], "b": [dt.time(1, 1, 1)]}
    expected = FrozenSchema({"a": nw.Struct({"a": nw.Datetime()}), "b": TIME})
    df = dataframe(data)
    assert expected == FrozenSchema(df)
    assert expected is FrozenSchema(df)
    assert expected is FrozenSchema(dataframe(deepcopy(data)))


def test_schema_hash() -> None:
    mapping = {"a": nw.List(F32), "b": nw.Struct({"a": STR, "c": BOOL})}
    schema_1 = FrozenSchema(mapping)
    schema_2 = FrozenSchema(**mapping)

    hash_1 = hash(schema_1)
    hash_2 = hash(schema_2)
    assert hash_1 == hash_2


def test_schema_empty() -> None:
    assert FrozenSchema() is FrozenSchema()


@pytest.mark.parametrize(
    ("iterable", "kwargs"),
    [
        ({"a": STR, "b": I64}, {"b": F64}),
        ([("b", BOOL), ("c", I32)], {"d": DATE}),
        ((("b", BOOL), ("c", I32)), {"c": DATE}),
        ({"c": DATE, "b": I32}, {"b": BINARY, "B": I32}),
        (FrozenSchema(inception=TIME), {"also_this": DATE}),
    ],
    ids=[
        "dict-override",
        "list-extend",
        "tuple-override",
        "dict-override-extend",
        "frozen-merge",
    ],
)
def test_schema_new_merge(
    iterable: IntoSchema | FrozenSchema, kwargs: dict[str, DType]
) -> None:
    result = FrozenSchema(iterable, **kwargs)
    expected = dict(iterable, **kwargs)
    assert result.items() == expected.items()


@pytest.mark.parametrize("function", [copy, deepcopy], ids=["copy", "deepcopy"])
def test_schema_copy(function: Callable[[FrozenSchema], FrozenSchema]) -> None:
    # See https://github.com/narwhals-dev/narwhals/blob/41d8c8e06240b8cdfbfb85082ce8a73bdc5fae12/tests/plan/immutable_test.py#L129-L136
    schema = FrozenSchema({"a": DATE, "b": STR, "c": BINARY})
    clone = function(schema)
    assert clone == schema
    assert clone is schema


def test_schema_setattr_delattr__() -> None:
    mapping = {"a": DATE, "b": STR, "c": BINARY}
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
