from __future__ import annotations

import pytest

import narwhals as nw
from narwhals._plan.schema import FrozenSchema, freeze_schema
from tests.plan.utils import dataframe


def test_schema() -> None:
    mapping = {"a": nw.Int64(), "b": nw.String()}
    schema = nw.Schema(mapping)
    frozen_schema = freeze_schema(mapping)

    assert frozen_schema.keys() == schema.keys()
    assert tuple(frozen_schema.values()) == tuple(schema.values())

    # NOTE: Would type-check if `Schema.__init__` didn't make liskov unhappy
    assert schema == nw.Schema(frozen_schema)  # type: ignore[arg-type]
    assert mapping == dict(frozen_schema)

    assert frozen_schema == freeze_schema(mapping)
    assert frozen_schema == freeze_schema(**mapping)
    assert frozen_schema == freeze_schema(a=nw.Int64(), b=nw.String())
    assert frozen_schema == freeze_schema(schema)
    assert frozen_schema == freeze_schema(frozen_schema)
    assert frozen_schema == freeze_schema(frozen_schema.items())

    # NOTE: Using `**` unpacking, despite not inheriting from `Mapping` or `dict`
    assert frozen_schema == freeze_schema(**frozen_schema)

    # NOTE: Using `HasSchema`
    df = dataframe({"a": [1, 2, 3], "b": ["c", "d", "e"]})
    assert frozen_schema == freeze_schema(df)

    # NOTE: In case this all looks *too good* to be true
    assert frozen_schema != freeze_schema(**mapping, c=nw.Float64())

    assert frozen_schema["a"] == schema["a"]

    assert frozen_schema.get("c") is None
    assert frozen_schema.get("c", nw.Unknown) is nw.Unknown
    assert frozen_schema.get("c", nw.Unknown()) == nw.Unknown()

    assert "b" in frozen_schema
    assert "e" not in frozen_schema

    with pytest.raises(TypeError, match="Cannot subclass 'FrozenSchema'"):

        class MutableSchema(FrozenSchema): ...  # type: ignore[misc]
