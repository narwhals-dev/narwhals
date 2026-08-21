from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

data = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]}


def test_cast(constructor: Constructor) -> None:
    to_dtypes = {"a": nw.Float64, "b": nw.Int32}
    df = nw.from_native(constructor(data))
    original = df.collect_schema()

    result = df.cast(to_dtypes)
    schema = result.collect_schema()
    assert schema == {**to_dtypes, "c": original["c"]}
    assert_equal_data(
        result, {"a": [1.0, 2.0, 3.0], "b": [4, 5, 6], "c": ["x", "y", "z"]}
    )


def test_cast_empty_mapping(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.cast({})
    assert result.collect_schema() == df.collect_schema()
    assert_equal_data(result, data)


def test_cast_nonexistent_column(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    with pytest.raises(nw.exceptions.ColumnNotFoundError):
        df.cast({"z": nw.Int64})


def test_cast_preserves_arrow_schema() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    field_a = pa.field("a", pa.int64(), nullable=False, metadata={b"k": b"v"})
    field_b = pa.field("b", pa.float64(), nullable=False)
    schema = pa.schema([field_a, field_b])
    schema_metadata = {b"pandas": b"{}"}
    # NOTE: `b` is declared non-nullable yet holds a null,
    # which `pa.Table.cast` rejects even when only `a` is being cast.
    native = pa.Table.from_arrays(
        [pa.array([1, 2]), pa.array([1.0, None])], schema=schema
    ).replace_schema_metadata(schema_metadata)
    result = nw.from_native(native, eager_only=True).cast({"a": nw.Int32}).to_native()
    out_schema = result.schema
    assert out_schema.metadata == schema_metadata
    assert out_schema.field("a") == pa.field(
        "a", pa.int32(), nullable=field_a.nullable, metadata=field_a.metadata
    )
    assert out_schema.field("b") == field_b


def test_cast_invalid_raises_narwhals_error() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame(data), eager_only=True)
    with pytest.raises(nw.exceptions.InvalidOperationError):
        df.cast({"c": nw.Int64})
