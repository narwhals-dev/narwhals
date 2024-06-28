from datetime import datetime
from datetime import timezone
from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw

data = {
    "a": [datetime(2020, 1, 1)],
    "b": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
}


def test_schema_comparison() -> None:
    assert {"a": nw.String()} != {"a": nw.Int32()}
    assert {"a": nw.Int32()} == {"a": nw.Int32()}


def test_object() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]}).astype(object)
    result = nw.from_native(df).schema
    assert result["a"] == nw.Object


def test_string_disguised_as_object() -> None:
    df = pd.DataFrame({"a": ["foo", "bar"]}).astype(object)
    result = nw.from_native(df).schema
    assert result["a"] == nw.String


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_actual_object(constructor: Any) -> None:
    class Foo: ...

    data = {"a": [Foo()]}
    df = nw.from_native(constructor(data))
    result = df.schema
    assert result == {"a": nw.Object}
