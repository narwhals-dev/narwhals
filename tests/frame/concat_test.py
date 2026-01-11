from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from narwhals.schema import Schema
from tests.utils import Constructor, ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import LazyFrameT


def _cast(frame: LazyFrameT, schema: Schema) -> LazyFrameT:
    return frame.select(nw.col(name).cast(dtype) for name, dtype in schema.items())


def test_concat_horizontal(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_left = nw.from_native(constructor_eager(data), eager_only=True)

    data_right = {"c": [6, 12, -1], "d": [0, -4, 2]}
    df_right = nw.from_native(constructor_eager(data_right), eager_only=True)

    result = nw.concat([df_left, df_right], how="horizontal")
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "c": [6, 12, -1],
        "d": [0, -4, 2],
    }
    assert_equal_data(result, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([])
    pattern = re.compile(r"horizontal.+not supported.+lazyframe", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nw.concat([df_left.lazy()], how="horizontal")


def test_concat_vertical(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_left = (
        nw.from_native(constructor(data)).lazy().rename({"a": "c", "b": "d"}).drop("z")
    )

    data_right = {"c": [6, 12, -1], "d": [0, -4, 2]}
    df_right = nw.from_native(constructor(data_right)).lazy()

    result = nw.concat([df_left, df_right], how="vertical")
    expected = {"c": [1, 3, 2, 6, 12, -1], "d": [4, 4, 6, 0, -4, 2]}
    assert_equal_data(result, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([], how="vertical")

    with pytest.raises(
        (Exception, TypeError),
        match=r"unable to vstack|inputs should all have the same schema",
    ):
        nw.concat([df_left, df_right.rename({"d": "i"})], how="vertical").collect()
    with pytest.raises(
        (Exception, TypeError),
        match=r"unable to vstack|unable to append|inputs should all have the same schema",
    ):
        nw.concat([df_left, df_left.select("d")], how="vertical").collect()


def test_concat_diagonal(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "ibis" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data_1 = {"a": [1, 3], "b": [4, 6]}
    data_2 = {"a": [100, 200], "z": ["x", "y"]}
    expected = {
        "a": [1, 3, 100, 200],
        "b": [4, 6, None, None],
        "z": [None, None, "x", "y"],
    }

    df_1 = nw.from_native(constructor(data_1)).lazy()
    df_2 = nw.from_native(constructor(data_2)).lazy()

    result = nw.concat([df_1, df_2], how="diagonal")

    assert_equal_data(result, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([], how="diagonal")


@pytest.mark.parametrize(
    ("ldata", "lschema", "rdata", "rschema", "expected_data", "expected_schema"),
    [
        (
            {"a": [1, 2, 3], "b": [True, False, None]},
            Schema({"a": nw.Int8(), "b": nw.Boolean()}),
            {"a": [43, 2, 3], "b": [32, 1, None]},
            Schema({"a": nw.Int16(), "b": nw.Int64()}),
            {"a": [1, 2, 3, 43, 2, 3], "b": [1, 0, None, 32, 1, None]},
            Schema({"a": nw.Int16(), "b": nw.Int64()}),
        ),
        (
            {"a": [1, 2], "b": [2, 1]},
            Schema({"a": nw.Int32(), "b": nw.Int32()}),
            {"a": [1.0, 0.2], "b": [None, 0.1]},
            Schema({"a": nw.Float32(), "b": nw.Float32()}),
            {"a": [1.0, 2.0, 1.0, 0.2], "b": [2.0, 1.0, None, 0.1]},
            Schema({"a": nw.Float64(), "b": nw.Float64()}),
        ),
    ],
    ids=["nullable-integer", "nullable-float"],
)
def test_concat_vertically_relaxed(
    constructor: Constructor,
    ldata: dict[str, Any],
    lschema: Schema,
    rdata: dict[str, Any],
    rschema: Schema,
    expected_data: dict[str, Any],
    expected_schema: Schema,
    request: pytest.FixtureRequest,
) -> None:
    # Adapted from https://github.com/pola-rs/polars/blob/b0fdbd34d430d934bda9a4ca3f75e136223bd95b/py-polars/tests/unit/functions/test_concat.py#L64
    is_nullable_int = request.node.callspec.id.endswith("nullable-integer")
    if is_nullable_int and any(
        x in str(constructor)
        for x in ("dask", "pandas_constructor", "modin_constructor", "cudf")
    ):
        reason = "Cannot convert non-finite values (NA or inf)"
        request.applymarker(pytest.mark.xfail(reason=reason))
    left = nw.from_native(constructor(ldata)).lazy().pipe(_cast, lschema)
    right = nw.from_native(constructor(rdata)).lazy().pipe(_cast, rschema)
    result = nw.concat([left, right], how="vertical_relaxed")

    assert result.collect_schema() == expected_schema
    assert_equal_data(result.collect(), expected_data)

    result = nw.concat([right, left], how="vertical_relaxed")
    assert result.collect_schema() == expected_schema


@pytest.mark.parametrize(
    ("schema1", "schema2", "schema3", "expected_schema"),
    [
        (
            Schema({"a": nw.Int32(), "c": nw.Int64()}),
            Schema({"a": nw.Float64(), "b": nw.Float32()}),
            Schema({"b": nw.Int32(), "c": nw.Int32()}),
            Schema({"a": nw.Float64(), "c": nw.Int64(), "b": nw.Float64()}),
        ),
        (
            Schema({"a": nw.Float32(), "c": nw.Float32()}),
            Schema({"a": nw.Float64(), "b": nw.Float32()}),
            Schema({"b": nw.Float32(), "c": nw.Float32()}),
            Schema({"a": nw.Float64(), "c": nw.Float32(), "b": nw.Float32()}),
        ),
    ],
    ids=["nullable-integer", "nullable-float"],
)
def test_concat_diagonal_relaxed(
    constructor: Constructor,
    schema1: Schema,
    schema2: Schema,
    schema3: Schema,
    expected_schema: Schema,
    request: pytest.FixtureRequest,
) -> None:
    # Adapted from https://github.com/pola-rs/polars/blob/b0fdbd34d430d934bda9a4ca3f75e136223bd95b/py-polars/tests/unit/functions/test_concat.py#L265C1-L288C41
    is_nullable_int = request.node.callspec.id.endswith("nullable-integer")
    if is_nullable_int and any(
        x in str(constructor)
        for x in ("dask", "pandas_constructor", "modin_constructor", "cudf")
    ):
        reason = "Cannot convert non-finite values (NA or inf)"
        request.applymarker(pytest.mark.xfail(reason=reason))

    if "ibis" in str(constructor):
        pytest.skip(reason="NotImplementedError")

    data1 = {"a": [1, 2], "c": [10, 20]}
    df1 = nw.from_native(constructor(data1)).lazy().pipe(_cast, schema1)

    data2 = {"a": [3.5, 4.5], "b": [30.1, 40.2]}
    df2 = nw.from_native(constructor(data2)).lazy().pipe(_cast, schema2)

    data3 = {"b": [5, 6], "c": [50, 60]}
    df3 = nw.from_native(constructor(data3)).lazy().pipe(_cast, schema3)

    result = nw.concat([df1, df2, df3], how="diagonal_relaxed")
    out_schema = result.collect_schema()
    assert out_schema == expected_schema

    expected_data = {
        "a": [1.0, 2.0, 3.5, 4.5, None, None],
        "c": [10, 20, None, None, 50, 60],
        "b": [None, None, 30.1, 40.2, 5.0, 6.0],
    }
    assert_equal_data(result.collect(), expected_data)
