from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import DASK_VERSION, Constructor, ConstructorEager, assert_equal_data


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
    [("__and__", [True, False, False, False]), ("__or__", [True, True, True, False])],
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


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("__and__", [False, False, False, False]),
        ("__rand__", [False, False, False, False]),
        ("__or__", [True, True, False, False]),
        ("__ror__", [True, True, False, False]),
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
        and DASK_VERSION < (2024, 10)
        and operator in {"__rand__", "__ror__"}
    ):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [True, True, False, False]}
    df = nw.from_native(constructor(data))

    result = df.select(a=getattr(nw.col("a"), operator)(False))  # noqa: FBT003
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
