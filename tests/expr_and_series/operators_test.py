from __future__ import annotations

from typing import Literal

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import DASK_VERSION, Constructor, ConstructorEager, assert_equal_data


@pytest.mark.parametrize(
    "operator", ["__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__"]
)
@pytest.mark.parametrize("mask", [0, 1, 2**63 - 1, 2**63, 2**64 - 1])
def test_bitwise_uint64_scalar_pyarrow(operator: str, mask: int) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    values = [2**63, 2**64 - 1, None]
    native = pa.table(
        {
            "a": pa.array(values, type=pa.uint64()),
            "mask": pa.array([mask] * len(values), type=pa.uint64()),
        }
    )
    df = nw.from_native(native, eager_only=True)
    expected = [getattr(v, operator)(mask) if v is not None else None for v in values]
    series = getattr(df["a"], operator)(mask)
    assert_equal_data({"a": series}, {"a": expected})
    assert series.to_native().type == pa.uint64()
    result = df.select(a=getattr(nw.col("a"), operator)(nw.col("mask").first()))
    assert_equal_data(result, {"a": expected})
    assert result.to_native()["a"].type == pa.uint64()
    if mask <= 2**63 - 1:
        result = df.select(a=getattr(nw.col("a"), operator)(mask))
        assert_equal_data(result, {"a": expected})
        result = df.select(a=getattr(nw.lit(mask), operator)(nw.col("a")))
        assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    "operator", ["__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__"]
)
def test_bitwise_uint64_signed_scalar_pyarrow(operator: str) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    df = nw.from_native(pa.table({"a": pa.array([2**63, None], type=pa.uint64())}))
    mask = nw.lit(None, dtype=nw.Int64)
    assert_equal_data(
        df.select(a=getattr(nw.col("a"), operator)(mask)), {"a": [None, None]}
    )
    assert_equal_data(
        df.select(a=getattr(mask, operator)(nw.col("a"))), {"a": [None, None]}
    )
    with pytest.raises(pa.ArrowInvalid):
        getattr(df["a"], operator)(-1)
    with pytest.raises(pa.ArrowInvalid):
        df.select(a=getattr(nw.lit(-1), operator)(nw.col("a")))
    with pytest.raises(pa.ArrowNotImplementedError):
        getattr(df["a"], operator)(1.5)


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("__eq__", [False, True, False]),
        ("__ne__", [True, False, True]),
        ("__le__", [True, True, False]),
        ("__lt__", [True, False, False]),
        ("__ge__", [False, True, True]),
        ("__gt__", [False, False, True]),
    ],
)
def test_comparand_operators_scalar_expr(
    constructor: Constructor, operator: str, expected: list[bool]
) -> None:
    data = {"a": [0, 1, 2]}
    df = nw.from_native(constructor(data))
    result = df.select(getattr(nw.col("a"), operator)(1))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("__eq__", [True, False, False]),
        ("__ne__", [False, True, True]),
        ("__le__", [True, False, True]),
        ("__lt__", [False, False, True]),
        ("__ge__", [True, True, False]),
        ("__gt__", [False, True, False]),
    ],
)
def test_comparand_operators_expr(
    constructor: Constructor, operator: str, expected: list[bool]
) -> None:
    data = {"a": [0, 1, 1], "b": [0, 0, 2]}
    df = nw.from_native(constructor(data))
    result = df.select(getattr(nw.col("a"), operator)(nw.col("b")))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("__and__", [True, False, False, False]),
        ("__or__", [True, True, True, False]),
        ("__xor__", [False, True, True, False]),
    ],
)
def test_logic_operators_expr(
    constructor: Constructor, operator: str, expected: list[bool]
) -> None:
    data = {"a": [True, True, False, False], "b": [True, False, True, False]}
    df = nw.from_native(constructor(data))

    result = df.select(getattr(nw.col("a"), operator)(nw.col("b")))
    assert_equal_data(result, {"a": expected})


def test_logic_operators_expr_kleene(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor):
        # https://github.com/rapidsai/cudf/issues/19171
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        # Dask infers `[True, None, None, None]` as `object` dtype, and then `__or__` fails.
        request.applymarker(pytest.mark.xfail)
    data = {"a": [True, True, False, None], "b": [True, None, None, None]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a") | (nw.col("b")))
    if any(x in str(constructor) for x in ("pandas_constructor",)):
        expected: list[bool | None] = [True, True, False, False]
    else:
        expected = [True, True, None, None]
    assert_equal_data(result, {"a": expected})
    result = df.select(nw.col("a") & (nw.col("b")))
    if any(x in str(constructor) for x in ("pandas_constructor",)):
        expected = [True, False, False, False]
    else:
        expected = [True, None, False, None]
    assert_equal_data(result, {"a": expected})
    result = df.select(nw.col("a") ^ (nw.col("b")))
    if any(x in str(constructor) for x in ("pandas_constructor",)):
        expected = [False, True, False, False]
    else:
        expected = [False, None, None, None]
    assert_equal_data(result, {"a": expected})


def test_xor_operator_integers(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "sqlframe", "ibis", "pyspark")):
        # PySpark raises, DuckDB and SQLFrame cast to boolean,
        # Ibis compiles `~(a & b)` on integers to `~a & b`.
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 2, 3], "b": [3, 2, 0]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a") ^ nw.col("b"), c=nw.col("a") ^ 1)
    assert_equal_data(result, {"a": [2, 0, 3], "c": [0, 3, 2]})


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
)
@pytest.mark.parametrize(
    ("operator", "expected", "expected_scalar"),
    [
        ("__and__", [1, 2, 0, None, None], [1, 0, 1, None, 0]),
        ("__rand__", [1, 2, 0, None, None], [1, 0, 1, None, 0]),
        ("__or__", [3, 2, 3, None, None], [1, 3, 3, None, 1]),
        ("__ror__", [3, 2, 3, None, None], [1, 3, 3, None, 1]),
        ("__xor__", [2, 0, 3, None, None], [0, 3, 2, None, 1]),
        ("__rxor__", [2, 0, 3, None, None], [0, 3, 2, None, 1]),
    ],
)
def test_bitwise_operators_pyarrow(
    dtype: Literal[
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"
    ],
    operator: str,
    expected: list[int | None],
    expected_scalar: list[int | None],
) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    native_dtype = pa.type_for_alias(dtype)
    native = pa.Table.from_arrays(
        [
            pa.chunked_array(
                [
                    pa.array([1, 2]).cast(native_dtype),
                    pa.array([3, None, 0]).cast(native_dtype),
                ]
            ),
            pa.chunked_array(
                [
                    pa.array([3, 2, 0]).cast(native_dtype),
                    pa.array([1, None]).cast(native_dtype),
                ]
            ),
        ],
        names=["a", "b"],
    )
    assert [len(chunk) for chunk in native["a"].chunks] == [2, 3]
    assert [len(chunk) for chunk in native["b"].chunks] == [3, 2]
    df = nw.from_native(native, eager_only=True)
    series = df["a"]
    result_series = getattr(series, operator)(df["b"])
    assert_equal_data({"a": result_series}, {"a": expected})
    assert result_series.to_native().type == native["a"].type
    result_expr = df.select(a=getattr(nw.col("a"), operator)(nw.col("b")))
    assert_equal_data(result_expr, {"a": expected})
    assert result_expr.to_native()["a"].type == native["a"].type
    assert_equal_data({"a": getattr(series, operator)(1)}, {"a": expected_scalar})
    result_scalar = df.select(a=getattr(nw.col("a"), operator)(1))
    assert_equal_data(result_scalar, {"a": expected_scalar})
    result_broadcast = df.select(a=getattr(nw.col("a"), operator)(nw.lit(1)))
    assert_equal_data(result_broadcast, {"a": expected_scalar})


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("__and__", [False, False, False, False]),
        ("__rand__", [False, False, False, False]),
        ("__or__", [True, True, False, False]),
        ("__ror__", [True, True, False, False]),
        ("__xor__", [True, True, False, False]),
        ("__rxor__", [True, True, False, False]),
    ],
)
def test_logic_operators_expr_scalar(
    constructor: Constructor,
    operator: str,
    expected: list[bool],
    request: pytest.FixtureRequest,
) -> None:
    if (
        "dask" in str(constructor)
        and DASK_VERSION < PkgVersion("2024.10")
        and operator in {"__rand__", "__ror__", "__rxor__"}
    ):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [True, True, False, False]}
    df = nw.from_native(constructor(data))

    result = df.select(a=getattr(nw.col("a"), operator)(False))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("__eq__", [False, True, False]),
        ("__ne__", [True, False, True]),
        ("__le__", [True, True, False]),
        ("__lt__", [True, False, False]),
        ("__ge__", [False, True, True]),
        ("__gt__", [False, False, True]),
    ],
)
def test_comparand_operators_scalar_series(
    constructor_eager: ConstructorEager, operator: str, expected: list[bool]
) -> None:
    data = {"a": [0, 1, 2]}
    s = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = {"a": (getattr(s, operator)(1))}
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("__eq__", [True, False, False]),
        ("__ne__", [False, True, True]),
        ("__le__", [True, False, True]),
        ("__lt__", [False, False, True]),
        ("__ge__", [True, True, False]),
        ("__gt__", [False, True, False]),
    ],
)
def test_comparand_operators_series(
    constructor_eager: ConstructorEager, operator: str, expected: list[bool]
) -> None:
    data = {"a": [0, 1, 1], "b": [0, 0, 2]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    series, other = df["a"], df["b"]
    result = {"a": getattr(series, operator)(other)}
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("__and__", [True, False, False, False]),
        ("__rand__", [True, False, False, False]),
        ("__or__", [True, True, True, False]),
        ("__ror__", [True, True, True, False]),
        ("__xor__", [False, True, True, False]),
        ("__rxor__", [False, True, True, False]),
    ],
)
def test_logic_operators_series(
    constructor_eager: ConstructorEager, operator: str, expected: list[bool]
) -> None:
    data = {"a": [True, True, False, False], "b": [True, False, True, False]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    series, other = df["a"], df["b"]
    result = {"a": getattr(series, operator)(other)}
    assert_equal_data(result, {"a": expected})
