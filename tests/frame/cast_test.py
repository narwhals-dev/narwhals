from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals.typing import IntoDType

data = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]}


def test_cast(constructor: Constructor) -> None:
    dtypes = {"a": nw.Float64, "b": nw.Int32}
    df = nw.from_native(constructor(data))
    schema = df.collect_schema()

    result = df.cast(dtypes)
    assert result.collect_schema() == {**dtypes, "c": schema["c"]}
    assert_equal_data(
        result, {"a": [1.0, 2.0, 3.0], "b": [4, 5, 6], "c": ["x", "y", "z"]}
    )


def test_cast_matches_expr_cast(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"t": [datetime(2020, 1, 1)], "a": [1], "s": [2]}))
    if not any(x in str(constructor) for x in ("pyspark", "sqlframe")):
        # NOTE: Spark-like backends only support microsecond precision.
        df = df.with_columns(nw.col("t").cast(nw.Datetime("ns")))
    dtypes: Mapping[str, IntoDType] = {"t": nw.Datetime, "a": nw.Int64, "s": nw.String}

    result = df.cast(dtypes)
    expected = df.with_columns(nw.col(k).cast(v) for k, v in dtypes.items())
    assert result.collect_schema() == expected.collect_schema()
    assert_equal_data(result, {"t": [datetime(2020, 1, 1)], "a": [1], "s": ["2"]})


def test_cast_invalid_dtype(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    with pytest.raises(TypeError, match="Expected Narwhals dtype"):
        df.cast({"a": "int64"})  # type: ignore[dict-item]


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

    metadata = {b"k": b"v"}
    field_a = pa.field("a", pa.int64(), nullable=False, metadata=metadata)
    field_b = pa.field("b", pa.float64(), nullable=False, metadata=metadata)
    fields: list[pa.Field[Any]] = [field_a, field_b]
    schema_metadata = {b"pandas": b"{}"}
    # NOTE: `b` is declared non-nullable yet holds a null,
    # which `pa.Table.cast` rejects even when only `a` is being cast.
    native = pa.Table.from_arrays(
        [pa.array([1, 2]), pa.array([1.0, None])], schema=pa.schema(fields)
    ).replace_schema_metadata(schema_metadata)
    result = nw.from_native(native, eager_only=True).cast({"a": nw.Int32}).to_native()
    schema = result.schema
    assert schema.metadata == schema_metadata
    assert schema.field("a").equals(
        pa.field("a", pa.int32(), nullable=False), check_metadata=True
    )
    assert schema.field("b").equals(field_b, check_metadata=True)


def test_cast_invalid_raises_narwhals_error() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame(data), eager_only=True)
    with pytest.raises(nw.exceptions.NarwhalsError):
        df.cast({"c": nw.Int64})
