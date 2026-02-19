from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.exceptions import MultiOutputExpressionError
from tests.utils import (
    DUCKDB_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    uses_pyarrow_backend,
)

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
    pytest.importorskip("numpy")
    import numpy as np

    df = nw.from_native(constructor_eager(data))

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
    pytest.importorskip("numpy")
    import numpy as np

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


def test_when_then_broadcasting(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a").sum() > 1).then("c"))
    expected = {"c": [4.1, 5, 6]}
    assert_equal_data(result, expected)
    result = df.select(nw.when(nw.col("a").sum() > 1).then(1).otherwise("c"))
    expected = {"literal": [1, 1, 1]}
    assert_equal_data(result, expected)


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
    otherwise: nw.Expr | int | None,
    expected: list[int],
    constructor: Constructor,
    request: pytest.FixtureRequest,
) -> None:
    if "cudf" in str(constructor) and otherwise is None:
        reason = "cudf does not support mixed types"
        request.applymarker(pytest.mark.xfail(reason=reason))
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
    otherwise: nw.Expr | int | None,
    expected: list[int],
    constructor: Constructor,
    request: pytest.FixtureRequest,
) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    if "cudf" in str(constructor) and otherwise is None:
        reason = "cudf does not support mixed types"
        request.applymarker(pytest.mark.xfail(reason=reason))

    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    expr = nw.when(condition).then(then).otherwise(otherwise)
    result = df.with_columns(a_when=expr)
    assert_equal_data(result.select(nw.col("a_when")), {"a_when": expected})


def test_when_then_empty(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [-1]})).filter(nw.col("a") > 0)
    result = df.with_columns(nw.when(nw.col("a") == 1).then(nw.lit(1)).alias("new_col"))
    expected: dict[str, Any] = {"a": [], "new_col": []}
    assert_equal_data(result, expected)


def test_when_chain_basic_two_conditions(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3]}))

    result = df.select(
        nw.when(nw.col("a") == 1)
        .then(10)
        .when(nw.col("a") == 2)
        .then(20)
        .otherwise(30)
        .alias("result")
    )

    expected = {"result": [10, 20, 30]}
    assert_equal_data(result, expected)


def test_when_chain_three_conditions(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3, 4]}))

    result = df.select(
        nw.when(nw.col("a") == 1)
        .then(100)
        .when(nw.col("a") == 2)
        .then(200)
        .when(nw.col("a") == 3)
        .then(300)
        .otherwise(400)
        .alias("result")
    )

    expected = {"result": [100, 200, 300, 400]}
    assert_equal_data(result, expected)


def test_when_chain_multiple_conditions(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3, 4, 5]}))

    result = df.select(
        nw.when(nw.col("a") == 1)
        .then(10)
        .when(nw.col("a") == 2)
        .then(20)
        .when(nw.col("a") == 3)
        .then(30)
        .when(nw.col("a") == 4)
        .then(40)
        .otherwise(50)
        .alias("result")
    )

    expected = {"result": [10, 20, 30, 40, 50]}
    assert_equal_data(result, expected)


def test_when_chain_no_otherwise(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3, 4]}))

    result = df.select(
        nw.when(nw.col("a") == 1).then(10).when(nw.col("a") == 2).then(20).alias("result")
    )

    expected = {"result": [10, 20, None, None]}
    assert_equal_data(result, expected)


def test_when_chain_first_match_wins(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3, 5, 10]}))

    result = df.select(
        nw.when(nw.col("a") < 5)
        .then(100)  # Matches 1, 2, 3
        .when(nw.col("a") < 3)
        .then(200)  # Would match 1, 2 but first condition already matched
        .when(nw.col("a") < 2)
        .then(300)  # Would match 1 but first condition already matched
        .otherwise(400)
        .alias("result")
    )

    expected = {"result": [100, 100, 100, 400, 400]}
    assert_equal_data(result, expected)


def test_when_chain_all_conditions_false(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3]}))

    result = df.select(
        nw.when(nw.col("a") > 10)
        .then(10)
        .when(nw.col("a") > 20)
        .then(20)
        .when(nw.col("a") > 30)
        .then(30)
        .otherwise(999)
        .alias("result")
    )

    expected = {"result": [999, 999, 999]}
    assert_equal_data(result, expected)


def test_when_chain_mixed_value_types(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [10, 20, 30]}))

    result = df.select(
        nw.when(nw.col("a") == 1)
        .then(nw.col("b"))  # Column reference
        .when(nw.col("a") == 2)
        .then(100)  # Scalar literal
        .when(nw.col("a") == 3)
        .then(nw.col("b") * 2)  # Expression
        .otherwise(0)
        .alias("result")
    )

    expected = {"result": [10, 100, 60]}
    assert_equal_data(result, expected)


def test_when_chain_complex_conditions(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}))

    result = df.select(
        nw.when((nw.col("a") == 1) & (nw.col("b") == 10))
        .then(111)
        .when((nw.col("a") > 2) & (nw.col("b") < 40))
        .then(222)
        .when(nw.col("a") + nw.col("b") > 50)
        .then(333)
        .otherwise(444)
        .alias("result")
    )

    expected = {"result": [111, 444, 222, 444, 333]}
    assert_equal_data(result, expected)


def test_when_chain_with_nulls(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if uses_pyarrow_backend(constructor) or "pyarrow" in str(constructor):
        reason = (
            "PyArrow uses SQL-like null semantics: when condition is null, "
            "result is null (not the otherwise value). This is pre-existing "
            "PyArrow behavior, not a chaining limitation."
        )
        request.applymarker(pytest.mark.xfail(reason=reason))

    df = nw.from_native(constructor({"a": [1, None, 3, None, 5]}))

    result = df.select(
        nw.when(nw.col("a") == 1)
        .then(10)
        .when(nw.col("a") == 3)
        .then(30)
        .when(nw.col("a") == 5)
        .then(50)
        .otherwise(99)
        .alias("result")
    )

    # Null values don't match any condition, so they get otherwise value
    expected = {"result": [10, 99, 30, 99, 50]}
    assert_equal_data(result, expected)


def test_when_chain_expression_conditions(constructor: Constructor) -> None:
    if uses_pyarrow_backend(constructor) or "pyarrow" in str(constructor):
        reason = (
            "PyArrow backend doesn't support modulo operator. "
            "This is a pre-existing PyArrow arithmetic limitation."
        )
        pytest.skip(reason=reason)

    df = nw.from_native(constructor({"a": [1, 2, 3, 4, 5]}))

    result = df.select(
        nw.when(nw.col("a") % 2 == 0)
        .then(1000)  # Even numbers
        .when(nw.col("a") % 3 == 0)
        .then(2000)  # Odd multiples of 3
        .otherwise(3000)
        .alias("result")
    )

    # 1: odd, not div by 3 -> 3000
    # 2: even -> 1000
    # 3: odd, div by 3 -> 2000 (first match)
    # 4: even -> 1000
    # 5: odd, not div by 3 -> 3000
    # Note: 3 is divisible by both 2 (false) and 3 (true)
    expected = {"result": [3000, 1000, 2000, 1000, 3000]}
    assert_equal_data(result, expected)


def test_when_chain_value_types_string(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": ["x", "y", "z"]}))

    result = df.select(
        nw.when(nw.col("a") == 1)
        .then(nw.lit("first"))
        .when(nw.col("a") == 2)
        .then(nw.col("b"))
        .otherwise(nw.lit("other"))
        .alias("result")
    )

    expected = {"result": ["first", "y", "other"]}
    assert_equal_data(result, expected)


def test_when_chain_value_types_float(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}))

    result = df.select(
        nw.when(nw.col("a") == 1)
        .then(10.5)
        .when(nw.col("a") == 2)
        .then(nw.col("b"))
        .when(nw.col("a") == 3)
        .then(nw.col("b") * 2.0)
        .otherwise(0.0)
        .alias("result")
    )

    expected = {"result": [10.5, 2.2, 6.6]}
    assert_equal_data(result, expected)


def test_when_chain_value_types_mixed(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [10, 20, 30]}))

    result = df.select(
        nw.when(nw.col("a") == 1)
        .then(nw.col("b"))
        .when(nw.col("a") == 2)
        .then(100)
        .when(nw.col("a") == 3)
        .then(nw.col("b") * 2)
        .otherwise(0)
        .alias("result")
    )

    expected = {"result": [10, 100, 60]}
    assert_equal_data(result, expected)


def test_when_chain_with_columns(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [10, 20, 30]}))

    result = df.with_columns(
        nw.when(nw.col("a") == 1)
        .then(100)
        .when(nw.col("a") == 2)
        .then(200)
        .otherwise(300)
        .alias("c")
    )

    expected = {"a": [1, 2, 3], "b": [10, 20, 30], "c": [100, 200, 300]}
    assert_equal_data(result, expected)


def test_when_chain_multiple_columns(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3]}))

    result = df.select(
        nw.when(nw.col("a") == 1)
        .then(10)
        .when(nw.col("a") == 2)
        .then(20)
        .otherwise(30)
        .alias("x"),
        nw.when(nw.col("a") < 2)
        .then(100)
        .when(nw.col("a") < 3)
        .then(200)
        .otherwise(300)
        .alias("y"),
    )

    expected = {"x": [10, 20, 30], "y": [100, 200, 300]}
    assert_equal_data(result, expected)


def test_when_chain_backward_compatible_single_condition(
    constructor: Constructor,
) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3]}))

    # This is the existing API that must continue to work
    result = df.select(nw.when(nw.col("a") == 1).then(10).otherwise(20).alias("result"))

    expected = {"result": [10, 20, 20]}
    assert_equal_data(result, expected)


def test_when_chain_two_conditions_no_otherwise(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3, 4, 5]}))

    result = df.select(
        nw.when(nw.col("a") == 1).then(10).when(nw.col("a") == 2).then(20).alias("result")
    )

    expected = {"result": [10, 20, None, None, None]}
    assert_equal_data(result, expected)


def test_when_chain_conditions_on_multiple_columns(constructor: Constructor) -> None:
    df = nw.from_native(
        constructor({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40], "c": [5, 15, 25, 35]})
    )

    result = df.select(
        nw.when(nw.col("a") == 1)
        .then(nw.col("b"))
        .when(nw.col("c") > 20)
        .then(nw.col("c"))
        .otherwise(999)
        .alias("result")
    )

    # Row 0: a==1 -> b=10
    # Row 1: a!=1, c=15 not >20 -> 999
    # Row 2: a!=1, c=25 >20 -> 25
    # Row 3: a!=1, c=35 >20 -> 35
    expected = {"result": [10, 999, 25, 35]}
    assert_equal_data(result, expected)


def test_when_chain_boolean_column_condition(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [True, False, True, False], "b": [1, 2, 3, 4]}))

    result = df.select(
        nw.when(nw.col("a"))
        .then(100)
        .when(nw.col("b") > 2)
        .then(200)
        .otherwise(300)
        .alias("result")
    )

    # Row 0: a=True -> 100
    # Row 1: a=False, b=2 not >2 -> 300
    # Row 2: a=True -> 100
    # Row 3: a=False, b=4 >2 -> 200
    expected = {"result": [100, 300, 100, 200]}
    assert_equal_data(result, expected)
