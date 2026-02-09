from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.utils import (
    CUDF_VERSION,
    DASK_VERSION,
    PANDAS_VERSION,
    POLARS_VERSION,
    PYARROW_VERSION,
    Constructor,
    assert_equal_data,
    is_pyspark_connect,
)

if TYPE_CHECKING:
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType, PythonLiteral


@pytest.mark.parametrize(
    ("dtype", "expected_lit"),
    [(None, [2, 2, 2]), (nw.String, ["2", "2", "2"]), (nw.Float32, [2.0, 2.0, 2.0])],
)
def test_lit(
    constructor: Constructor, dtype: DType | None, expected_lit: list[Any]
) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()
    result = df.with_columns(nw.lit(2, dtype).alias("lit"))
    expected = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0], "lit": expected_lit}
    assert_equal_data(result, expected)


def test_lit_error(constructor: Constructor) -> None:
    pytest.importorskip("numpy")
    import numpy as np

    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()
    with pytest.raises(
        ValueError, match="numpy arrays are not supported as literal values"
    ):
        _ = df.with_columns(nw.lit(np.array([1, 2])).alias("lit"))  # pyright: ignore[reportArgumentType]


def test_lit_out_name(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()
    result = df.with_columns(nw.lit(2))
    expected = {"a": [1, 3, 2], "literal": [2, 2, 2]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("col_name", "expr", "expected_result"),
    [
        ("left_lit", nw.lit(1) + nw.col("a"), [2, 4, 3]),
        ("right_lit", nw.col("a") + nw.lit(1), [2, 4, 3]),
        ("right_lit_with_abs", nw.col("a") + nw.lit(-1).abs(), [2, 4, 3]),
        ("left_lit_with_agg", nw.lit(1) + nw.col("a").mean(), [3]),
        ("right_lit_with_agg", nw.col("a").mean() - nw.lit(1), [1]),
        ("left_scalar", 1 + nw.col("a"), [2, 4, 3]),
        ("right_scalar", nw.col("a") + 1, [2, 4, 3]),
        ("left_scalar_with_agg", 1 + nw.col("a").mean(), [3]),
        ("right_scalar_with_agg", nw.col("a").mean() - 1, [1]),
        ("lit_compare", nw.col("a") == nw.lit(3), [False, True, False]),
    ],
)
def test_lit_operation_in_select(
    constructor: Constructor, col_name: str, expr: nw.Expr, expected_result: list[int]
) -> None:
    if (
        "dask" in str(constructor)
        and col_name == "right_lit_with_abs"
        and DASK_VERSION < (2025,)
    ):
        pytest.skip()

    data = {"a": [1, 3, 2]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()
    result = df.select(expr.alias(col_name))
    expected = {col_name: expected_result}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("col_name", "expr", "expected_result"),
    [
        ("lit_and_scalar", (nw.lit(2) + 1), [3, 3, 3]),
        ("scalar_and_lit", (1 + nw.lit(2)), [3, 3, 3]),
    ],
)
def test_lit_operation_in_with_columns(
    constructor: Constructor, col_name: str, expr: nw.Expr, expected_result: list[int]
) -> None:
    data = {"a": [1, 3, 2]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()
    result = df.with_columns(expr.alias(col_name))
    expected = {"a": data["a"], col_name: expected_result}
    assert_equal_data(result, expected)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_date_lit(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    # https://github.com/dask/dask/issues/11637
    if "dask" in str(constructor) or (
        # https://github.com/rapidsai/cudf/pull/18832
        "cudf" in str(constructor) and CUDF_VERSION >= (25, 8, 0)
    ):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor):
        pytest.importorskip("pyarrow")
    df = nw.from_native(constructor({"a": [1]}))
    result = df.with_columns(nw.lit(date(2020, 1, 1), dtype=nw.Date)).collect_schema()
    if df.implementation.is_cudf():
        # cudf has no date dtype
        assert result == {"a": nw.Int64, "literal": nw.Datetime}
    else:
        assert result == {"a": nw.Int64, "literal": nw.Date}


def test_pyarrow_lit_string() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    df = nw.from_native(pa.table({"a": [1, 2, 3]}))
    result = df.select(nw.lit("foo")).to_native().schema.field("literal")
    assert pa.types.is_string(result.type)
    result = df.select(nw.lit("foo", dtype=nw.String)).to_native().schema.field("literal")
    assert pa.types.is_string(result.type)


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        # Empty nested structures
        ((), nw.List(nw.Int32())),
        ([], nw.List(nw.Int32())),
        ({}, nw.Struct({})),
        # Nested structures with different size from dataframe
        (("foo", "bar"), None),
        (["orca", "narwhal"], None),
        ({"field_1": 42}, None),
        # Nested structures with same size as dataframe
        (("foo", "bar", "baz"), None),
        (["orca", "narwhal", "penguin"], None),
        (
            {"field_1": 42, "field_2": 1.2, "field_3": True},
            nw.Struct(
                {"field_1": nw.Int32(), "field_2": nw.Float64(), "field_3": nw.Boolean()}
            ),
        ),
    ],
)
def test_nested_structures(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    value: PythonLiteral,
    dtype: IntoDType | None,
) -> None:
    is_empty_dict = isinstance(value, dict) and len(value) == 0
    non_pyspark_sql_like = ("duckdb", "sqlframe", "ibis")
    is_non_pyspark_sql_like = any(x in str(constructor) for x in non_pyspark_sql_like)
    if (is_non_pyspark_sql_like or is_pyspark_connect(constructor)) and is_empty_dict:
        reason = "Cannot create an empty struct type for backend"
        request.applymarker(pytest.mark.xfail(reason=reason, raises=NotImplementedError))

    # TODO(FBruzzesi): Check cudf
    if any(x in str(constructor) for x in ("cudf", "dask")):
        reason = "Nested structures are not support for backend"
        request.applymarker(pytest.mark.xfail(reason=reason, raises=NotImplementedError))

    if any(x in str(constructor) for x in ("pandas", "modin")) and (
        PYARROW_VERSION == (0, 0, 0) or PANDAS_VERSION < (2, 0)
    ):  # pragma: no cover
        reason = "Requires pyarrow and pandas 2.0+"
        pytest.skip(reason=reason)

    if (
        "polars" in str(constructor)
        and isinstance(value, dict)
        and POLARS_VERSION < (1, 10, 0)
    ):  # pragma: no cover
        reason = "polars<1.10 does not support dict to struct in lit"
        pytest.skip(reason=reason)

    size = 3
    data = {"a": list(range(size))}
    expr = nw.lit(value, dtype=dtype).alias("nested")

    value_ = list(value) if isinstance(value, tuple) else value
    expected_nested = {"nested": [value_] * size}

    frame = nw.from_native(constructor(data))

    result_with_cols = frame.with_columns(expr)
    assert_equal_data(result_with_cols, {**data, **expected_nested})

    result_select = frame.select(expr, nw.col("a"))
    assert_equal_data(result_select, {**expected_nested, **data})


@pytest.mark.parametrize("value", [[], (), {}])
def test_raise_empty_nested_structures(value: PythonLiteral) -> None:
    msg = "Cannot infer dtype for empty nested structure. Please provide an explicit dtype parameter."
    with pytest.raises(ValueError, match=msg):
        nw.lit(value=value)


@pytest.mark.parametrize(
    "value",
    [
        # List containing nested structures
        [[1, 2], [3, 4]],
        [(1, 2), (3, 4)],
        [{"a": 1}, {"a": 2}],
        # Tuple containing nested structures
        ([1, 2], [3, 4]),
        ((1, 2), (3, 4)),
        ({"a": 1}, {"a": 2}),
        # Dict containing nested structures
        {"a": [1, 2], "b": [3, 4]},
        {"a": (1, 2), "b": (3, 4)},
        {"a": {"x": 1}, "b": {"y": 2}},
    ],
)
def test_raise_nested_structures_with_nested_values(value: Any) -> None:
    msg = "Nested structures with nested values are not supported."
    with pytest.raises(NotImplementedError, match=msg):
        nw.lit(value=value)
