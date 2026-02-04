from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [1, 2, 3]}


def test_expr_mean_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    if any(x in str(constructor) for x in ("pyarrow_table_constructor",)):
        with pytest.raises(NotImplementedError, match="Correlation"):
            df.select(corr=nw.corr(nw.col("a"), nw.col("b")))
        return
    result = df.select(corr=nw.corr(nw.col("a"), nw.col("b")))
    expected = {"corr": [0.5]}
    assert_equal_data(result, expected)


def test_expr_mean_expr_str(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    if any(x in str(constructor) for x in ("pyarrow_table_constructor",)):
        with pytest.raises(NotImplementedError, match="Correlation"):
            df.select(corr=nw.corr("a", "b"))
        return
    result = df.select(corr=nw.corr("a", "b"))
    expected = {"corr": [0.5]}
    assert_equal_data(result, expected)


def test_expr_mean_expr_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    if any(x in str(constructor_eager) for x in ("pyarrow_table_constructor",)):
        with pytest.raises(NotImplementedError, match="Correlation"):
            df.select(corr=nw.corr(df["a"], df["b"]))
        return
    result = df.select(corr=nw.corr(df["a"], df["b"]))
    expected = {"corr": [0.5]}
    assert_equal_data(result, expected)
