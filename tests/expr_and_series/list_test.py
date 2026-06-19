from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {
    "a": [1, 2, 3],
    "b": [4, 5, 6],
    "d": [[10, 20], [30], [40, 50, 60]],
    "e": [[100], [200, 300], [400, 500]],
}

UNSUPPORTED_BACKENDS = ("dask",)


def maybe_skip(constructor: Constructor | ConstructorEager) -> None:
    if "pandas" in str(constructor) and (
        PANDAS_VERSION < (2, 2, 0) or PYARROW_VERSION == (0, 0, 0)
    ):
        reason = "pandas is too old or pyarrow not installed"
        pytest.skip(reason=reason)


@pytest.mark.parametrize(
    "exprs", [(nw.col("a"), nw.col("b")), ([nw.col("a"), nw.col("b")]), ("a", "b")]
)
def test_list_positional_exprs(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    exprs: tuple[nw.Expr | list[nw.Expr], ...],
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)

    maybe_skip(constructor=constructor)

    df = nw.from_native(constructor(data))
    result = df.select(nw.list(*exprs).alias("list"))

    expected = {"list": [[1, 4], [2, 5], [3, 6]]}
    assert_equal_data(result, expected)


def test_list_single_column(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)

    maybe_skip(constructor=constructor)

    df = nw.from_native(constructor(data))
    result = df.select(nw.list("a").alias("list"))

    expected = {"list": [[1], [2], [3]]}
    assert_equal_data(result, expected)


def test_list_with_expressions(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)

    maybe_skip(constructor=constructor)

    df = nw.from_native(constructor(data))
    result = df.select(nw.list(nw.col("a") * 2, nw.col("b") + 1).alias("list"))

    expected = {"list": [[2, 5], [4, 6], [6, 7]]}
    assert_equal_data(result, expected)


def test_list_of_lists(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)

    maybe_skip(constructor=constructor)

    df = nw.from_native(constructor(data))
    result = df.select(nw.list("d", "e").alias("nested"))

    expected = {
        "nested": [[[10, 20], [100]], [[30], [200, 300]], [[40, 50, 60], [400, 500]]]
    }
    assert_equal_data(result, expected)


def test_list_raise_no_exprs(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    with pytest.raises(ValueError, match="expected at least 1 expression in 'list'"):
        df.select(nw.list().alias("list"))
