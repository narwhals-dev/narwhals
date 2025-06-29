from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError, MultiOutputExpressionError
from tests.utils import Constructor, ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import _1DArray

data = {
    "a": [1, 2, 3],
    "b": ["a", "b", "c"],
    "c": [4.1, 5.0, 6.0],
    "d": [True, False, True],
    "e": [7.0, 2.0, 1.1],
}


def test_when(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a") == 1).then(value=3).alias("a_when"))
    expected = {"a_when": [3, None, None]}
    assert_equal_data(result, expected)
    result = df.select(nw.when(nw.col("a") == 1).then(value=3))
    expected = {"literal": [3, None, None]}
    assert_equal_data(result, expected)


def test_when_otherwise(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a") == 1).then(3).otherwise(6).alias("a_when"))
    expected = {"a_when": [3, 6, 6]}
    assert_equal_data(result, expected)


def test_multiple_conditions(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.when(nw.col("a") < 3, nw.col("c") < 5.0).then(3).alias("a_when")
    )
    expected = {"a_when": [3, None, None]}
    assert_equal_data(result, expected)


def test_no_arg_when_fail(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    with pytest.raises((TypeError, ValueError)):
        df.select(nw.when().then(value=3).alias("a_when"))


def test_value_numpy_array(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    import numpy as np

    result = df.select(nw.when(nw.col("a") == 1).then(np.arange(3, 6)).alias("a_when"))
    expected = {"a_when": [3, None, None]}
    assert_equal_data(result, expected)


def test_value_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    s_data = {"s": [3, 4, 5]}
    s = nw.from_native(constructor_eager(s_data))["s"]
    assert isinstance(s, nw.Series)
    result = df.select(nw.when(nw.col("a") == 1).then(s).alias("a_when"))
    expected = {"a_when": [3, None, None]}
    assert_equal_data(result, expected)


def test_value_expression(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a") == 1).then(nw.col("a") + 9).alias("a_when"))
    expected = {"a_when": [10, None, None]}
    assert_equal_data(result, expected)


def test_otherwise_numpy_array(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))

    arr: _1DArray = np.zeros([3], np.dtype(np.int64))
    arr[:3] = 0, 9, 10
    result = df.select(nw.when(nw.col("a") == 1).then(-1).otherwise(arr).alias("a_when"))
    expected = {"a_when": [-1, 9, 10]}
    assert_equal_data(result, expected)


def test_otherwise_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    s_data = {"s": [0, 9, 10]}
    s = nw.from_native(constructor_eager(s_data))["s"]
    assert isinstance(s, nw.Series)
    result = df.select(nw.when(nw.col("a") == 1).then(-1).otherwise(s).alias("a_when"))
    expected = {"a_when": [-1, 9, 10]}
    assert_equal_data(result, expected)


def test_otherwise_expression(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.when(nw.col("a") == 1).then(-1).otherwise(nw.col("a") + 7).alias("a_when")
    )
    expected = {"a_when": [-1, 9, 10]}
    assert_equal_data(result, expected)


def test_when_then_otherwise_into_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a") > 1).then("c").otherwise("e"))
    expected = {"c": [7, 5, 6]}
    assert_equal_data(result, expected)


def test_when_then_invalid(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    with pytest.raises(InvalidOperationError):
        df.select(nw.when(nw.col("a").sum() > 1).then("c"))

    with pytest.raises(InvalidOperationError):
        df.select(nw.when(nw.col("a").sum() > 1).then(1).otherwise("c"))


def test_when_then_otherwise_lit_str(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a") > 1).then(nw.col("b")).otherwise(nw.lit("z")))
    expected = {"b": ["z", "b", "c"]}
    assert_equal_data(result, expected)


def test_when_then_otherwise_both_lit(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(
        x1=nw.when(nw.col("a") > 1).then(nw.lit(42)).otherwise(nw.lit(-1)),
        x2=nw.when(nw.col("a") > 2).then(nw.lit(42)).otherwise(nw.lit(-1)),
    )
    expected = {"x1": [-1, 42, 42], "x2": [-1, -1, 42]}
    assert_equal_data(result, expected)


def test_when_then_otherwise_multi_output(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    with pytest.raises(MultiOutputExpressionError):
        df.select(x1=nw.when(nw.all() > 1).then(nw.col("a", "b")))
    with pytest.raises(MultiOutputExpressionError):
        df.select(x1=nw.when(nw.all() > 1).then(nw.lit(1)).otherwise(nw.all()))


@pytest.mark.parametrize(
    ("condition", "then", "otherwise", "expected"),
    [
        (nw.col("a").sum() == 6, 100, None, [100]),
        (nw.col("a").sum() == 6, 100, 200, [100]),
        (nw.col("a").sum() == 6, nw.col("a").sum(), 200, [6]),
        (nw.col("a").sum() == 6, 100, nw.col("b").sum(), [100]),
        (nw.col("a").sum() == 6, nw.col("a").sum(), nw.col("b").sum(), [6]),
        (nw.col("a").sum() == 5, 100, None, [None]),
        (nw.col("a").sum() == 5, 100, 200, [200]),
        (nw.col("a").sum() == 5, nw.col("a").sum(), 200, [200]),
        (nw.col("a").sum() == 5, 100, nw.col("b").sum(), [15]),
        (nw.col("a").sum() == 5, nw.col("a").sum(), nw.col("b").sum(), [15]),
    ],
)
def test_when_then_otherwise_aggregate_select(
    condition: nw.Expr,
    then: nw.Expr | int,
    otherwise: nw.Expr | int,
    expected: list[int],
    constructor: Constructor,
) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = df.select(a_when=nw.when(condition).then(then).otherwise(otherwise))
    assert_equal_data(result, {"a_when": expected})


@pytest.mark.parametrize(
    ("condition", "then", "otherwise", "expected"),
    [
        (nw.col("a").sum() == 6, 100, None, [100, 100, 100]),
        (nw.col("a").sum() == 6, 100, 200, [100, 100, 100]),
        (nw.col("a").sum() == 6, nw.col("a").sum(), 200, [6, 6, 6]),
        (nw.col("a").sum() == 6, 100, nw.col("b").sum(), [100, 100, 100]),
        (nw.col("a").sum() == 6, nw.col("a").sum(), nw.col("b").sum(), [6, 6, 6]),
        (nw.col("a").sum() == 5, 100, None, [None, None, None]),
        (nw.col("a").sum() == 5, 100, 200, [200, 200, 200]),
        (nw.col("a").sum() == 5, nw.col("a").sum(), 200, [200, 200, 200]),
        (nw.col("a").sum() == 5, 100, nw.col("b").sum(), [15, 15, 15]),
        (nw.col("a").sum() == 5, nw.col("a").sum(), nw.col("b").sum(), [15, 15, 15]),
    ],
)
def test_when_then_otherwise_aggregate_with_columns(
    condition: nw.Expr,
    then: nw.Expr | int,
    otherwise: nw.Expr | int,
    expected: list[int],
    constructor: Constructor,
) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    expr = nw.when(condition).then(then).otherwise(otherwise)
    result = df.with_columns(a_when=expr)
    assert_equal_data(result.select(nw.col("a_when")), {"a_when": expected})
