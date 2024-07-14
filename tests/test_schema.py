from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw

data = {"a": nw.Int64(), "b": nw.Float32(), "c": nw.String()}


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("names", ["a", "b", "c"]),
        ("dtypes", [nw.Int64(), nw.Float32(), nw.String()]),
        ("len", 3),
    ],
)
def test_schema_object(method: str, expected: Any) -> None:
    schema = nw.Schema(data)
    assert getattr(schema, method)() == expected
