from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


@pytest.mark.parametrize(
    ("attr", "rhs", "expected"),
    [
        ("__add__", 1, [2, 3, 4]),
        ("__sub__", 1, [0, 1, 2]),
        ("__mul__", 2, [2, 4, 6]),
        ("__truediv__", 2.0, [0.5, 1.0, 1.5]),
        ("__truediv__", 1, [1, 2, 3]),
        ("__floordiv__", 2, [0, 1, 1]),
        ("__mod__", 2, [1, 0, 1]),
        ("__pow__", 2, [1, 4, 9]),
    ],
)
def test_arithmetic(
    attr: str,
    rhs: Any,
    expected: list[Any],
    constructor: Any,
    request: Any,
) -> None:
    if "dask" in str(constructor) and attr not in [
        "__add__",
        "__sub__",
        "__mul__",
    ]:
        request.applymarker(pytest.mark.xfail)
    if attr == "__mod__" and any(
        x in str(constructor) for x in ["pandas_pyarrow", "pyarrow_table", "modin"]
    ):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1.0, 2, 3]}
    df = nw.from_native(constructor(data))
    result = df.select(getattr(nw.col("a"), attr)(rhs))
    compare_dicts(result, {"a": expected})


@pytest.mark.parametrize(
    ("attr", "rhs", "expected"),
    [
        ("__radd__", 1, [2, 3, 4]),
        ("__rsub__", 1, [0, -1, -2]),
        ("__rmul__", 2, [2, 4, 6]),
        ("__rtruediv__", 2.0, [2, 1, 2 / 3]),
        ("__rfloordiv__", 2, [2, 1, 0]),
        ("__rmod__", 2, [0, 0, 2]),
        ("__rpow__", 2, [2, 4, 8]),
    ],
)
def test_right_arithmetic(
    attr: str,
    rhs: Any,
    expected: list[Any],
    constructor: Any,
    request: Any,
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if attr == "__rmod__" and any(
        x in str(constructor) for x in ["pandas_pyarrow", "pyarrow_table", "modin"]
    ):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor(data))
    result = df.select(a=getattr(nw.col("a"), attr)(rhs))
    compare_dicts(result, {"a": expected})


@pytest.mark.parametrize(
    ("attr", "rhs", "expected"),
    [
        ("__add__", 1, [2, 3, 4]),
        ("__sub__", 1, [0, 1, 2]),
        ("__mul__", 2, [2, 4, 6]),
        ("__truediv__", 2.0, [0.5, 1.0, 1.5]),
        ("__truediv__", 1, [1, 2, 3]),
        ("__floordiv__", 2, [0, 1, 1]),
        ("__mod__", 2, [1, 0, 1]),
        ("__pow__", 2, [1, 4, 9]),
    ],
)
def test_arithmetic_series(
    attr: str,
    rhs: Any,
    expected: list[Any],
    constructor_eager: Any,
    request: Any,
) -> None:
    if attr == "__mod__" and any(
        x in str(constructor_eager) for x in ["pandas_pyarrow", "pyarrow_table", "modin"]
    ):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(getattr(df["a"], attr)(rhs))
    compare_dicts(result, {"a": expected})


@pytest.mark.parametrize(
    ("attr", "rhs", "expected"),
    [
        ("__radd__", 1, [2, 3, 4]),
        ("__rsub__", 1, [0, -1, -2]),
        ("__rmul__", 2, [2, 4, 6]),
        ("__rtruediv__", 2.0, [2, 1, 2 / 3]),
        ("__rfloordiv__", 2, [2, 1, 0]),
        ("__rmod__", 2, [0, 0, 2]),
        ("__rpow__", 2, [2, 4, 8]),
    ],
)
def test_right_arithmetic_series(
    attr: str,
    rhs: Any,
    expected: list[Any],
    constructor_eager: Any,
    request: Any,
) -> None:
    if attr == "__rmod__" and any(
        x in str(constructor_eager) for x in ["pandas_pyarrow", "pyarrow_table", "modin"]
    ):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(a=getattr(df["a"], attr)(rhs))
    compare_dicts(result, {"a": expected})
