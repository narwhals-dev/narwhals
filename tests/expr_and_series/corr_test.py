from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [1, 2, 3]}


@pytest.mark.parametrize(
    ("output_name", "a", "b", "expected_corr"),
    [
        ("a", "a", "b", 0.5),
        ("b", nw.col("a").alias("b"), nw.col("b").alias("c"), 0.5),
        ("a", nw.col("a") * nw.col("b"), nw.col("b") - 0.5, 0.8660254037844386),
    ],
)
def test_corr_expr(
    constructor: Constructor,
    output_name: str,
    a: str | nw.Expr,
    b: str | nw.Expr,
    expected_corr: float,
    request: pytest.FixtureRequest,
) -> None:
    df = nw.from_native(constructor(data))
    if any(x in str(constructor) for x in ("pyarrow_table_constructor",)):
        # not implemented yet
        request.applymarker(pytest.mark.xfail)
    result = df.select(nw.corr(a, b))
    expected = {output_name: [expected_corr]}
    assert_equal_data(result, expected)
    result = df.select(result=nw.corr(a, b))
    expected = {"result": [expected_corr]}
    assert_equal_data(result, expected)


def test_corr_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    if any(x in str(constructor_eager) for x in ("pyarrow_table_constructor",)):
        with pytest.raises(NotImplementedError, match="Correlation"):
            df.select(corr=nw.corr(df["a"], df["b"]))
        return
    result = df.select(corr=nw.corr(df["a"], df["b"]))
    expected = {"corr": [0.5]}
    assert_equal_data(result, expected)
