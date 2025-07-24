from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pandas as pd
import pyarrow as pa
import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import (
    DUCKDB_VERSION,
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {
    "a": ["a", "a", "b", "b", "b"],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
    "i": [0, 1, 2, 3, 4],
}

data_cum = {
    "a": ["a", "a", "b", "b", "b"],
    "b": [1, 2, None, 5, 3],
    "c": [5, 4, 3, 2, 1],
    "i": [0, 1, 2, 3, 4],
}


def test_over_single(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()

    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "i": list(range(5)),
        "c_max": [5, 5, 3, 3, 3],
    }

    result = df.with_columns(c_max=nw.col("c").max().over("a")).sort("i")
    assert_equal_data(result, expected)
    result = df.with_columns(c_max=nw.col("c").max().over(["a"])).sort("i")
    assert_equal_data(result, expected)


def test_over_std_var(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "cudf" in str(constructor):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()

    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "i": list(range(5)),
        "c_std0": [0.5, 0.5, 0.816496580927726, 0.816496580927726, 0.816496580927726],
        "c_std1": [0.7071067811865476, 0.7071067811865476, 1.0, 1.0, 1.0],
        "c_var0": [
            0.25,
            0.25,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
        ],
        "c_var1": [0.5, 0.5, 1.0, 1.0, 1.0],
    }

    result = df.with_columns(
        c_std0=nw.col("c").std(ddof=0).over("a"),
        c_std1=nw.col("c").std(ddof=1).over("a"),
        c_var0=nw.col("c").var(ddof=0).over("a"),
        c_var1=nw.col("c").var(ddof=1).over("a"),
    ).sort("i")
    assert_equal_data(result, expected)


def test_over_multiple(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "i": list(range(5)),
        "c_min": [5, 4, 1, 2, 1],
    }

    result = df.with_columns(c_min=nw.col("c").min().over("a", "b")).sort("i")
    assert_equal_data(result, expected)


def test_over_cumsum(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2, 1):
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor_eager):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data_cum))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cumsum": [1, 3, None, 5, 8],
        "c_cumsum": [5, 9, 3, 5, 6],
    }

    result = (
        df.with_columns(nw.col("b", "c").cum_sum().over("a").name.suffix("_cumsum"))
        .sort("i")
        .drop("i")
    )
    assert_equal_data(result, expected)


def test_over_cumcount(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor_eager):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data_cum))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cumcount": [1, 2, 0, 1, 2],
        "c_cumcount": [1, 2, 1, 2, 3],
    }

    result = (
        df.with_columns(nw.col("b", "c").cum_count().over("a").name.suffix("_cumcount"))
        .sort("i")
        .drop("i")
    )
    assert_equal_data(result, expected)


def test_over_cummax(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2, 1):
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor_eager):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data_cum))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cummax": [1, 2, None, 5, 5],
        "c_cummax": [5, 5, 3, 3, 3],
    }
    result = (
        df.with_columns(nw.col("b", "c").cum_max().over("a").name.suffix("_cummax"))
        .sort("i")
        .drop("i")
    )
    assert_equal_data(result, expected)


def test_over_cummin(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2, 1):
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor_eager):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data_cum))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cummin": [1, 1, None, 5, 3],
        "c_cummin": [5, 4, 3, 2, 1],
    }

    result = (
        df.with_columns(nw.col("b", "c").cum_min().over("a").name.suffix("_cummin"))
        .sort("i")
        .drop("i")
    )
    assert_equal_data(result, expected)


def test_over_cumprod(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2, 1):
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor_eager):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data_cum))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cumprod": [1, 2, None, 5, 15],
        "c_cumprod": [5, 20, 3, 6, 6],
    }

    result = (
        df.with_columns(nw.col("b", "c").cum_prod().over("a").name.suffix("_cumprod"))
        .sort("i")
        .drop("i")
    )
    assert_equal_data(result, expected)


def test_over_anonymous_cumulative(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor_eager):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager({"": [1, 1, 2], "b": [4, 5, 6]}))
    context = (
        pytest.raises(NotImplementedError)
        if df.implementation.is_pyarrow()
        else pytest.raises(KeyError)  # type: ignore[arg-type]
        if df.implementation.is_modin()
        or (df.implementation.is_pandas() and PANDAS_VERSION < (1, 3))
        # TODO(unassigned): bug in old pandas + modin.
        # df.groupby('a')[['a', 'b']].cum_sum() excludes `'a'` from result
        else does_not_raise()
    )
    with context:
        result = df.with_columns(
            nw.all().cum_sum().over("").name.suffix("_cum_sum")
        ).sort("", "b")
        expected = {
            "": [1, 1, 2],
            "b": [4, 5, 6],
            "_cum_sum": [1, 2, 2],
            "b_cum_sum": [4, 9, 6],
        }
        assert_equal_data(result, expected)


def test_over_anonymous_reduction(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    if "modin" in str(constructor):
        # probably bugged
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor({"a": [1, 1, 2], "b": [4, 5, 6]}))
    context = (
        pytest.raises(NotImplementedError)
        if df.implementation.is_pyarrow()
        else does_not_raise()
    )
    with context:
        result = (
            nw.from_native(df)
            .with_columns(nw.all().sum().over("a").name.suffix("_sum"))
            .sort("a", "b")
        )
        expected = {
            "a": [1, 1, 2],
            "b": [4, 5, 6],
            "a_sum": [2, 2, 2],
            "b_sum": [9, 9, 6],
        }
        assert_equal_data(result, expected)


def test_over_unsupported() -> None:
    dfpd = pd.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    with pytest.raises(NotImplementedError):
        nw.from_native(dfpd).select(nw.col("a").null_count().over("a"))


def test_over_unsupported_dask() -> None:
    pytest.importorskip("dask")
    import dask.dataframe as dd

    df = dd.from_pandas(pd.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]}))
    with pytest.raises(NotImplementedError):
        nw.from_native(df).select(nw.col("a").null_count().over("a"))


def test_over_shift(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor_eager):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_shift": [None, None, None, None, 3],
    }
    result = df.with_columns(b_shift=nw.col("b").shift(2).over("a")).sort("i").drop("i")
    assert_equal_data(result, expected)


def test_over_diff(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor_eager):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_diff": [None, 1, None, 2, -2],
    }
    result = df.with_columns(b_diff=nw.col("b").diff().over("a")).sort("i").drop("i")
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("attr", "expected_b"),
    [
        ("cum_max", [5, 5, 9, None, 9]),
        ("cum_min", [4, 5, 7, None, 9]),
        ("cum_sum", [9, 5, 16, None, 9]),
        ("cum_count", [2, 1, 2, 1, 1]),
        ("cum_prod", [20, 5, 63, None, 9]),
    ],
)
def test_over_cum_reverse(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    attr: str,
    expected_b: list[object],
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if "pandas_nullable" in str(constructor_eager) and attr in {"cum_max", "cum_min"}:
        # https://github.com/pandas-dev/pandas/issues/61031
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor_eager):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)
    df = constructor_eager({"a": [1, 1, 2, 2, 2], "b": [4, 5, 7, None, 9]})
    expr = getattr(nw.col("b"), attr)(reverse=True)
    result = nw.from_native(df).with_columns(expr.over("a"))
    expected = {"a": [1, 1, 2, 2, 2], "b": expected_b}
    assert_equal_data(result, expected)


def test_over_raise_len_change(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))

    with pytest.raises(InvalidOperationError):
        nw.from_native(df).select(nw.col("b").drop_nulls().over("a"))


def test_unsupported_over() -> None:
    data = {"a": [1, 2, 3, 4, 5, 6], "b": ["x", "x", "x", "y", "y", "y"]}
    df = pd.DataFrame(data)
    with pytest.raises(NotImplementedError, match="elementary"):
        nw.from_native(df).select(nw.col("a").shift(1).cum_sum().over("b"))
    tbl = pa.table(data)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="aggregation or literal"):
        nw.from_native(tbl).select(nw.col("a").shift(1).cum_sum().over("b"))


def test_over_without_partition_by(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        # windows not yet supported
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor({"a": [1, -1, 2], "i": [0, 2, 1]}))
    result = (
        df.with_columns(b=nw.col("a").abs().cum_sum().over(order_by="i"))
        .sort("i")
        .select("a", "b", "i")
    )
    expected = {"a": [1, 2, -1], "b": [1, 3, 4], "i": [0, 1, 2]}
    assert_equal_data(result, expected)


def test_len_over_2369(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    if "pandas" in str(constructor) and PANDAS_VERSION < (1, 5):
        pytest.skip()
    if any(x in str(constructor) for x in ("modin", "cudf")):
        # https://github.com/modin-project/modin/issues/7508
        # https://github.com/rapidsai/cudf/issues/18491
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 2, 4], "b": ["x", "x", "y"]}))
    result = df.with_columns(a_len_per_group=nw.len().over("b")).sort("a")
    expected = {"a": [1, 2, 4], "b": ["x", "x", "y"], "a_len_per_group": [2, 2, 1]}
    assert_equal_data(result, expected)


def test_over_quantile(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "pyspark", "cudf")):
        # cudf: https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3, 4, 5, 6], "b": ["x", "x", "x", "y", "y", "y"]}

    quantile_expr = nw.col("a").quantile(quantile=0.5, interpolation="linear")
    native_frame = constructor(data)

    if "dask" in str(constructor):
        native_frame = native_frame.repartition(npartitions=1)  # type: ignore[union-attr]

    result = (
        nw.from_native(native_frame)
        .with_columns(
            quantile_over_b=quantile_expr.over("b"), quantile_global=quantile_expr
        )
        .sort("a")
    )

    expected = {
        **data,
        "quantile_over_b": [2, 2, 2, 5, 5, 5],
        "quantile_global": [3.5] * 6,
    }
    assert_equal_data(result, expected)


def test_over_ewm_mean(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor_eager) for x in ("pyarrow_table", "modin", "cudf")):
        # not implemented
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor_eager) and PANDAS_VERSION < (1, 2):
        request.applymarker(pytest.mark.xfail(reason="too old, not implemented"))

    data = {"a": [0.0, 1.0, 3.0, 5.0, 7.0, 7.5], "b": [1, 1, 1, 2, 2, 2]}

    ewm_expr = nw.col("a").ewm_mean(com=1)
    result = (
        nw.from_native(constructor_eager(data))
        .with_columns(ewm_over_b=ewm_expr.over("b"), ewm_global=ewm_expr)
        .sort("a")
    )
    expected = {
        **data,
        "ewm_over_b": [0.0, 2 / 3, 2.0, 5.0, 6 + 1 / 3, 7.0],
        "ewm_global": [0.0, 2 / 3, 2.0, 3.6, 5.354838709677419, 6.444444444444445],
    }
    assert_equal_data(result, expected)
