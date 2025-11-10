"""Tricks to generate nan's and null's for pandas with nullable backends.

* Square rooting a negative number will generate a NaN
* Replacing a value with None once the dtype is nullable will generate <NA>'s
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.exceptions import ComputeError, InvalidOperationError
from tests.conftest import (
    dask_lazy_p1_constructor,
    dask_lazy_p2_constructor,
    modin_constructor,
    pandas_constructor,
)
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import NumericLiteral

NON_NULLABLE_CONSTRUCTORS = (
    pandas_constructor,
    dask_lazy_p1_constructor,
    dask_lazy_p2_constructor,
    modin_constructor,
)
NULL_PLACEHOLDER, NAN_PLACEHOLDER = 9999.0, -1.0
INF_POS, INF_NEG = float("inf"), float("-inf")

data: dict[str, Any] = {
    "x": [1.001, NULL_PLACEHOLDER, NAN_PLACEHOLDER, INF_POS, INF_NEG, INF_POS],
    "y": [1.005, NULL_PLACEHOLDER, NAN_PLACEHOLDER, INF_POS, 3.0, INF_NEG],
    "non_numeric": list("number"),
    "idx": list(range(6)),
}


# Exceptions
def test_is_close_series_raise_non_numeric(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    x, y = df["non_numeric"], df["y"]

    msg = "`is_close` operation not supported for dtype"
    with pytest.raises(InvalidOperationError, match=msg):
        x.is_close(y)


@pytest.mark.parametrize("rel_tol", [1e-09, 999])
def test_is_close_raise_negative_abs_tol(
    constructor_eager: ConstructorEager, rel_tol: float
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    x, y = df["x"], df["y"]

    abs_tol = -2
    msg = rf"`abs_tol` must be non-negative but got {abs_tol}"
    with pytest.raises(ComputeError, match=msg):
        x.is_close(y, abs_tol=abs_tol, rel_tol=rel_tol)

    with pytest.raises(ComputeError, match=msg):
        df.select(nw.col("x").is_close(nw.col("y"), abs_tol=abs_tol, rel_tol=rel_tol))


@pytest.mark.parametrize("rel_tol", [-0.0001, 1.0, 1.1])
def test_is_close_raise_invalid_rel_tol(
    constructor_eager: ConstructorEager, rel_tol: float
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    x, y = df["x"], df["y"]

    msg = rf"`rel_tol` must be in the range \[0, 1\) but got {rel_tol}"
    with pytest.raises(ComputeError, match=msg):
        x.is_close(y, rel_tol=rel_tol)

    with pytest.raises(ComputeError, match=msg):
        df.select(nw.col("x").is_close(nw.col("y"), rel_tol=rel_tol))


cases_columnar = pytest.mark.parametrize(
    ("abs_tol", "rel_tol", "nans_equal", "expected"),
    [
        (0.1, 0.0, False, [True, None, False, True, False, False]),
        (0.0001, 0.0, True, [False, None, True, True, False, False]),
        (0.0, 0.1, False, [True, None, False, True, False, False]),
        (0.0, 0.001, True, [False, None, True, True, False, False]),
    ],
)
cases_scalar = pytest.mark.parametrize(
    ("other", "abs_tol", "rel_tol", "nans_equal", "expected"),
    [
        (1.0, 0.1, 0.0, False, [True, None, False, False, False, False]),
        (1.0, 0.0001, 0.0, True, [False, None, False, False, False, False]),
        (2.9, 0.0, 0.1, False, [False, None, False, False, True, False]),
        (2.9, 0.0, 0.001, True, [False, None, False, False, False, False]),
    ],
)


# Series
@cases_columnar
def test_is_close_series_with_series(
    constructor_eager: ConstructorEager,
    abs_tol: float,
    rel_tol: float,
    *,
    nans_equal: bool,
    expected: list[Any],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    x, y = df["x"], df["y"]
    nulls = nw.new_series(
        "nulls", [None] * len(x), nw.Float64(), backend=df.implementation
    )
    x = x.zip_with(x != NAN_PLACEHOLDER, x**0.5).zip_with(x != NULL_PLACEHOLDER, nulls)
    y = y.zip_with(y != NAN_PLACEHOLDER, y**0.5).zip_with(y != NULL_PLACEHOLDER, nulls)
    result = x.is_close(y, abs_tol=abs_tol, rel_tol=rel_tol, nans_equal=nans_equal)

    if constructor_eager in NON_NULLABLE_CONSTRUCTORS:
        expected = [v if v is not None else nans_equal for v in expected]
    elif "pandas" in str(constructor_eager) and PANDAS_VERSION >= (3,):
        expected = [
            v if data["y"][i] not in {NULL_PLACEHOLDER, NAN_PLACEHOLDER} else None
            for i, v in enumerate(expected)
        ]
    assert_equal_data({"result": result}, {"result": expected})


@cases_scalar
def test_is_close_series_with_scalar(
    constructor_eager: ConstructorEager,
    other: NumericLiteral,
    abs_tol: float,
    rel_tol: float,
    *,
    nans_equal: bool,
    expected: list[Any],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    y = df["y"]
    nulls = nw.new_series(
        "nulls", [None] * len(y), nw.Float64(), backend=df.implementation
    )
    y = y.zip_with(y != NAN_PLACEHOLDER, y**0.5).zip_with(y != NULL_PLACEHOLDER, nulls)
    result = y.is_close(other, abs_tol=abs_tol, rel_tol=rel_tol, nans_equal=nans_equal)

    if constructor_eager in NON_NULLABLE_CONSTRUCTORS:
        expected = [v if v is not None else False for v in expected]
    elif "pandas" in str(constructor_eager) and PANDAS_VERSION >= (3,):
        expected = [
            v if data["y"][i] not in {NULL_PLACEHOLDER, NAN_PLACEHOLDER} else None
            for i, v in enumerate(expected)
        ]
    assert_equal_data({"result": result}, {"result": expected})


# Expr
@cases_columnar
def test_is_close_expr_with_expr(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    abs_tol: float,
    rel_tol: float,
    *,
    nans_equal: bool,
    expected: list[Any],
) -> None:
    if "sqlframe" in str(constructor):
        # TODO(FBruzzesi): Figure out a MRE and report upstream
        reason = (
            "duckdb.duckdb.ParserException: Parser Error: syntax error at or near '='"
        )
        request.applymarker(pytest.mark.xfail(reason=reason))

    x, y = nw.col("x"), nw.col("y")
    result = (
        nw.from_native(constructor(data))
        .with_columns(
            x=nw.when(x != NAN_PLACEHOLDER).then(x).otherwise(x**0.5),
            y=nw.when(y != NAN_PLACEHOLDER).then(y).otherwise(y**0.5),
        )
        .with_columns(
            x=nw.when(x != NULL_PLACEHOLDER).then(x),
            y=nw.when(y != NULL_PLACEHOLDER).then(y),
        )
        .select(
            "idx",
            result=x.is_close(y, abs_tol=abs_tol, rel_tol=rel_tol, nans_equal=nans_equal),
        )
        .sort("idx")
    )
    if constructor in NON_NULLABLE_CONSTRUCTORS:
        expected = [v if v is not None else nans_equal for v in expected]
    elif "pandas" in str(constructor) and PANDAS_VERSION >= (3,):
        expected = [
            v if data["y"][i] not in {NULL_PLACEHOLDER, NAN_PLACEHOLDER} else None
            for i, v in enumerate(expected)
        ]
    assert_equal_data(result, {"idx": data["idx"], "result": expected})


@cases_scalar
def test_is_close_expr_with_scalar(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    other: NumericLiteral,
    abs_tol: float,
    rel_tol: float,
    *,
    nans_equal: bool,
    expected: list[Any],
) -> None:
    if "sqlframe" in str(constructor):
        # TODO(FBruzzesi): Figure out a MRE and report upstream
        reason = (
            "duckdb.duckdb.ParserException: Parser Error: syntax error at or near '='"
        )
        request.applymarker(pytest.mark.xfail(reason=reason))

    y = nw.col("y")
    result = (
        nw.from_native(constructor(data))
        .with_columns(y=nw.when(y != NAN_PLACEHOLDER).then(y).otherwise(y**0.5))
        .with_columns(y=nw.when(y != NULL_PLACEHOLDER).then(y))
        .select(
            "idx",
            result=y.is_close(
                other, abs_tol=abs_tol, rel_tol=rel_tol, nans_equal=nans_equal
            ),
        )
        .sort("idx")
    )
    if constructor in NON_NULLABLE_CONSTRUCTORS:
        expected = [v if v is not None else False for v in expected]
    elif "pandas" in str(constructor) and PANDAS_VERSION >= (3,):
        expected = [
            v if data["y"][i] not in {NULL_PLACEHOLDER, NAN_PLACEHOLDER} else None
            for i, v in enumerate(expected)
        ]
    assert_equal_data(result, {"idx": data["idx"], "result": expected})


def test_is_close_pandas_unnamed() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    ser = nw.from_native(pd.Series([1.1, 1.2]), series_only=True)
    res = ser.is_close(ser)
    assert res.name is None
    ser = nw.from_native(pd.Series([1.1, 1.2], name="ab"), series_only=True)
    res = ser.is_close(ser)
    assert res.name == "ab"
