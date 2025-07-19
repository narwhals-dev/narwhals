from __future__ import annotations

import datetime as dt
import os
import re
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
import pytest

import narwhals as nw
from narwhals.exceptions import ComputeError, DuplicateError, InvalidOperationError
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals.typing import NonNestedLiteral


data: Mapping[str, Any] = {"a": [1, 1, 3], "b": [4, 4, 6], "c": [7.0, 8.0, 9.0]}

df_pandas = pd.DataFrame(data)

POLARS_COLLECT_STREAMING_ENGINE = os.environ.get("NARWHALS_POLARS_NEW_STREAMING", None)


def test_group_by_complex() -> None:
    expected = {"a": [1, 3], "b": [-3.5, -3.0]}

    df = nw.from_native(df_pandas)
    with pytest.warns(UserWarning, match="complex group-by"):
        result_pd = nw.to_native(
            df.group_by("a").agg((nw.col("b") - nw.col("c").mean()).mean()).sort("a")
        )
    assert_equal_data(result_pd, expected)
    with pytest.raises(ValueError, match="complex aggregation"):
        nw.from_native(pa.table({"a": [1, 1, 2], "b": [4, 5, 6]})).group_by("a").agg(
            (nw.col("b") - nw.col("c").mean()).mean()
        )


def test_group_by_complex_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    expected = {"a": [1, 3], "b": [-3.5, -3.0]}

    df_lazy = pl.LazyFrame(data)
    lf = nw.from_native(df_lazy).lazy()
    result_pl = lf.group_by("a").agg((nw.col("b") - nw.col("c").mean()).mean()).sort("a")
    assert_equal_data(result_pl, expected)


def test_invalid_group_by_dask() -> None:
    pytest.importorskip("dask")
    import dask.dataframe as dd

    df_dask = dd.from_pandas(df_pandas)

    with pytest.raises(ValueError, match=r"Non-trivial complex aggregation found"):
        nw.from_native(df_dask).group_by("a").agg(nw.col("b").abs().min())


def test_group_by_iter(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    expected_keys = [(1,), (3,)]
    keys = []
    for key, sub_df in df.group_by("a"):
        if key == (1,):
            expected = {"a": [1, 1], "b": [4, 4], "c": [7.0, 8.0]}
            assert_equal_data(sub_df, expected)
            assert isinstance(sub_df, nw.DataFrame)
        keys.append(key)
    assert sorted(keys) == sorted(expected_keys)
    expected_keys = [(1, 4), (3, 6)]  # type: ignore[list-item]
    keys = []
    for key, _ in df.group_by("a", "b"):
        keys.append(key)
    assert sorted(keys) == sorted(expected_keys)
    keys = []
    for key, _ in df.group_by(["a", "b"]):
        keys.append(key)
    assert sorted(keys) == sorted(expected_keys)


def test_group_by_iter_non_str_pandas() -> None:
    expected = {"a": {0: [1], 1: ["a"]}, "b": {0: [2], 1: ["b"]}}
    df = nw.from_native(pd.DataFrame({0: [1, 2], 1: ["a", "b"]}))
    groups: dict[Any, Any] = {keys[0]: df for keys, df in df.group_by(1)}  # type: ignore[call-overload]
    assert groups.keys() == {"a", "b"}
    groups["a"] = groups["a"].to_dict(as_series=False)
    groups["b"] = groups["b"].to_dict(as_series=False)
    assert_equal_data(groups, expected)


def test_group_by_nw_all(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 1, 2], "b": [4, 5, 6], "c": [7, 8, 9]}))
    result = df.group_by("a").agg(nw.all().sum()).sort("a")
    expected = {"a": [1, 2], "b": [9, 6], "c": [15, 9]}
    assert_equal_data(result, expected)
    result = df.group_by("a").agg(nw.all().sum().name.suffix("_sum")).sort("a")
    expected = {"a": [1, 2], "b_sum": [9, 6], "c_sum": [15, 9]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("attr", "expected"),
    [
        ("sum", {"a": [1, 2], "b": [3, 3]}),
        ("mean", {"a": [1, 2], "b": [1.5, 3]}),
        ("max", {"a": [1, 2], "b": [2, 3]}),
        ("min", {"a": [1, 2], "b": [1, 3]}),
        ("std", {"a": [1, 2], "b": [0.707107, None]}),
        ("var", {"a": [1, 2], "b": [0.5, None]}),
        ("len", {"a": [1, 2], "b": [3, 1]}),
        ("n_unique", {"a": [1, 2], "b": [3, 1]}),
        ("count", {"a": [1, 2], "b": [2, 1]}),
    ],
)
def test_group_by_depth_1_agg(
    constructor: Constructor, attr: str, expected: dict[str, list[int | float]]
) -> None:
    if "pandas_pyarrow" in str(constructor) and attr == "var" and PANDAS_VERSION < (2, 1):
        pytest.skip(
            "Known issue with variance calculation in pandas 2.0.x with pyarrow backend in groupby operations"
        )
    data = {"a": [1, 1, 1, 2], "b": [1, None, 2, 3]}
    expr = getattr(nw.col("b"), attr)()
    result = nw.from_native(constructor(data)).group_by("a").agg(expr).sort("a")
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("attr", "ddof"), [("std", 0), ("var", 0), ("std", 2), ("var", 2)]
)
def test_group_by_depth_1_std_var(constructor: Constructor, attr: str, ddof: int) -> None:
    data = {"a": [1, 1, 1, 2, 2, 2], "b": [4, 5, 6, 0, 5, 5]}
    _pow = 0.5 if attr == "std" else 1
    expected = {
        "a": [1, 2],
        "b": [
            (sum((v - 5) ** 2 for v in [4, 5, 6]) / (3 - ddof)) ** _pow,
            (sum((v - 10 / 3) ** 2 for v in [0, 5, 5]) / (3 - ddof)) ** _pow,
        ],
    }
    expr = getattr(nw.col("b"), attr)(ddof=ddof)
    result = nw.from_native(constructor(data)).group_by("a").agg(expr).sort("a")
    assert_equal_data(result, expected)


def test_group_by_median(constructor: Constructor) -> None:
    data = {"a": [1, 1, 1, 2, 2, 2], "b": [5, 4, 6, 7, 3, 2]}
    result = (
        nw.from_native(constructor(data))
        .group_by("a")
        .agg(nw.col("b").median())
        .sort("a")
    )
    expected = {"a": [1, 2], "b": [5, 3]}
    assert_equal_data(result, expected)


def test_group_by_n_unique_w_missing(constructor: Constructor) -> None:
    data = {"a": [1, 1, 2], "b": [4, None, 5], "c": [None, None, 7], "d": [1, 1, 3]}
    result = (
        nw.from_native(constructor(data))
        .group_by("a")
        .agg(
            nw.col("b").n_unique(),
            c_n_unique=nw.col("c").n_unique(),
            c_n_min=nw.col("b").min(),
            d_n_unique=nw.col("d").n_unique(),
        )
        .sort("a")
    )
    expected = {
        "a": [1, 2],
        "b": [2, 1],
        "c_n_unique": [1, 1],
        "c_n_min": [4, 5],
        "d_n_unique": [1, 1],
    }
    assert_equal_data(result, expected)


def test_group_by_same_name_twice() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    pattern = re.compile(
        "expected unique.+names.+'b'.+2 times", re.IGNORECASE | re.DOTALL
    )
    with pytest.raises(DuplicateError, match=pattern):
        nw.from_native(df).group_by("a").agg(nw.col("b").sum(), nw.col("b").n_unique())


def test_group_by_empty_result_pandas() -> None:
    df_any = pd.DataFrame({"a": [1, 2, 3], "b": [4, 3, 2]})
    df = nw.from_native(df_any, eager_only=True)
    with pytest.raises(ValueError, match="No results"):
        df.filter(nw.col("a") < 0).group_by("a").agg(
            nw.col("b").sum().round(2).alias("c")
        )


def test_group_by_simple_named(constructor: Constructor) -> None:
    data = {"a": [1, 1, 2], "b": [4, 5, 6], "c": [7, 2, 1]}
    df = nw.from_native(constructor(data)).lazy()
    result = (
        df.group_by("a").agg(b_min=nw.col("b").min(), b_max=nw.col("b").max()).sort("a")
    )
    expected = {"a": [1, 2], "b_min": [4, 6], "b_max": [5, 6]}
    assert_equal_data(result, expected)


def test_group_by_simple_unnamed(constructor: Constructor) -> None:
    data = {"a": [1, 1, 2], "b": [4, 5, 6], "c": [7, 2, 1]}
    df = nw.from_native(constructor(data)).lazy()
    result = df.group_by("a").agg(nw.col("b").min(), nw.col("c").max()).sort("a")
    expected = {"a": [1, 2], "b": [4, 6], "c": [7, 1]}
    assert_equal_data(result, expected)


def test_group_by_multiple_keys(constructor: Constructor) -> None:
    data = {"a": [1, 1, 2], "b": [4, 4, 6], "c": [7, 2, 1]}
    df = nw.from_native(constructor(data)).lazy()
    result = (
        df.group_by("a", "b")
        .agg(c_min=nw.col("c").min(), c_max=nw.col("c").max())
        .sort("a")
    )
    expected = {"a": [1, 2], "b": [4, 6], "c_min": [2, 1], "c_max": [7, 1]}
    assert_equal_data(result, expected)


def test_key_with_nulls(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "modin" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="Modin flaky here", strict=False))

    data = {"b": [4, 5, None], "a": [1, 2, 3]}
    result = (
        nw.from_native(constructor(data))
        .group_by("b")
        .agg(nw.len(), nw.col("a").min())
        .sort("a")
        .with_columns(nw.col("b").cast(nw.Float64))
    )
    expected = {"b": [4.0, 5, None], "len": [1, 1, 1], "a": [1, 2, 3]}
    assert_equal_data(result, expected)


def test_key_with_nulls_ignored(constructor: Constructor) -> None:
    data = {"b": [4, 5, None], "a": [1, 2, 3]}
    result = (
        nw.from_native(constructor(data))
        .group_by("b", drop_null_keys=True)
        .agg(nw.len(), nw.col("a").min())
        .sort("a")
        .with_columns(nw.col("b").cast(nw.Float64))
    )
    expected = {"b": [4.0, 5], "len": [1, 1], "a": [1, 2]}
    assert_equal_data(result, expected)


def test_key_with_nulls_iter(constructor_eager: ConstructorEager) -> None:
    data = {
        "b": [None, "4", "5", None, "7"],
        "a": [None, 1, 2, 3, 4],
        "c": [None, "4", "3", None, None],
    }
    result = dict(
        nw.from_native(constructor_eager(data), eager_only=True)
        .group_by("b", "c", drop_null_keys=True)
        .__iter__()
    )

    assert len(result) == 2
    assert_equal_data(result[("4", "4")], {"b": ["4"], "a": [1], "c": ["4"]})
    assert_equal_data(result[("5", "3")], {"b": ["5"], "a": [2], "c": ["3"]})

    result = dict(
        nw.from_native(constructor_eager(data), eager_only=True)
        .group_by("b", "c", drop_null_keys=False)
        .__iter__()
    )
    assert_equal_data(result[("4", "4")], {"b": ["4"], "a": [1], "c": ["4"]})
    assert_equal_data(result[("5", "3")], {"b": ["5"], "a": [2], "c": ["3"]})
    assert len(result) == 4


def test_no_agg(constructor: Constructor) -> None:
    result = nw.from_native(constructor(data)).group_by(["a", "b"]).agg().sort("a", "b")

    expected = {"a": [1, 3], "b": [4, 6]}
    assert_equal_data(result, expected)


def test_group_by_categorical(constructor: Constructor) -> None:
    if any(x in str(constructor) for x in ("pyspark", "duckdb", "ibis")):
        pytest.skip(reason="no categorical support")
    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (
        15,
    ):  # pragma: no cover
        # https://github.com/narwhals-dev/narwhals/issues/1078
        pytest.skip(
            reason="The defaults for grouping by categories in pandas are different"
        )

    data = {"g1": ["a", "a", "b", "b"], "g2": ["x", "y", "x", "z"], "x": [1, 2, 3, 4]}
    df = nw.from_native(constructor(data))
    result = (
        df.with_columns(
            g1=nw.col("g1").cast(nw.Categorical()), g2=nw.col("g2").cast(nw.Categorical())
        )
        .group_by(["g1", "g2"])
        .agg(nw.col("x").sum())
        .sort("x")
    )
    assert_equal_data(result, data)


def test_group_by_shift_raises(constructor: Constructor) -> None:
    df_native = {"a": [1, 2, 3], "b": [1, 1, 2]}
    df = nw.from_native(constructor(df_native))
    with pytest.raises(InvalidOperationError, match="does not aggregate"):
        df.group_by("b").agg(nw.col("a").shift(1))


def test_double_same_aggregation(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("dask", "cudf")):
        # bugged in dask https://github.com/dask/dask/issues/11612
        # and cudf https://github.com/rapidsai/cudf/issues/17649
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 1, 2], "b": [4, 5, 6]}))
    result = df.group_by("a").agg(c=nw.col("b").mean(), d=nw.col("b").mean()).sort("a")
    expected = {"a": [1, 2], "c": [4.5, 6], "d": [4.5, 6]}
    assert_equal_data(result, expected)


def test_all_kind_of_aggs(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("dask", "cudf")):
        # bugged in dask https://github.com/dask/dask/issues/11612
        # and cudf https://github.com/rapidsai/cudf/issues/17649
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor) and PANDAS_VERSION < (1, 4):
        pytest.skip(
            "Pandas < 1.4.0 does not support multiple aggregations with the same column"
        )
    df = nw.from_native(constructor({"a": [1, 1, 1, 2, 2, 2], "b": [4, 5, 6, 0, 5, 5]}))
    result = (
        df.group_by("a")
        .agg(
            c=nw.col("b").mean(),
            d=nw.col("b").mean(),
            e=nw.col("b").std(ddof=1),
            f=nw.col("b").std(ddof=2),
            g=nw.col("b").var(ddof=2),
            h=nw.col("b").var(ddof=2),
            i=nw.col("b").n_unique(),
        )
        .sort("a")
    )

    variance_num = sum((v - 10 / 3) ** 2 for v in [0, 5, 5])
    expected = {
        "a": [1, 2],
        "c": [5, 10 / 3],
        "d": [5, 10 / 3],
        "e": [1, (variance_num / (3 - 1)) ** 0.5],
        "f": [2**0.5, (variance_num) ** 0.5],  # denominator is 1 (=3-2)
        "g": [2.0, variance_num],  # denominator is 1 (=3-2)
        "h": [2.0, variance_num],  # denominator is 1 (=3-2)
        "i": [3, 2],
    }
    assert_equal_data(result, expected)


def test_pandas_group_by_index_and_column_overlap() -> None:
    df = pd.DataFrame(
        {"a": [1, 1, 2], "b": [4, 5, 6]}, index=pd.Index([0, 1, 2], name="a")
    )
    result = nw.from_native(df, eager_only=True).group_by("a").agg(nw.col("b").mean())
    expected = {"a": [1, 2], "b": [4.5, 6.0]}
    assert_equal_data(result, expected)

    key, result = next(iter(nw.from_native(df, eager_only=True).group_by("a")))
    assert key == (1,)
    expected_native = pd.DataFrame({"a": [1, 1], "b": [4, 5]}, index=pd.Index([0, 1]))
    pd.testing.assert_frame_equal(result.to_native(), expected_native)


def test_fancy_functions(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 1, 2], "b": [4, 5, 6]}))
    result = df.group_by("a").agg(nw.all().std(ddof=0)).sort("a")
    expected = {"a": [1, 2], "b": [0.5, 0.0]}
    assert_equal_data(result, expected)
    result = df.group_by("a").agg(nw.selectors.numeric().std(ddof=0)).sort("a")
    assert_equal_data(result, expected)
    result = df.group_by("a").agg(nw.selectors.matches("b").std(ddof=0)).sort("a")
    assert_equal_data(result, expected)
    result = (
        df.group_by("a").agg(nw.selectors.matches("b").std(ddof=0).alias("c")).sort("a")
    )
    expected = {"a": [1, 2], "c": [0.5, 0.0]}
    assert_equal_data(result, expected)
    result = (
        df.group_by("a")
        .agg(nw.selectors.matches("b").std(ddof=0).name.map(lambda _x: "c"))
        .sort("a")
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("keys", "aggs", "expected", "sort_by"),
    [
        (
            [nw.col("a").abs(), nw.col("a").abs().alias("a_with_alias")],
            [nw.col("x").sum()],
            {"a": [1, 2], "a_with_alias": [1, 2], "x": [5, 5]},
            ["a"],
        ),
        (
            [nw.col("a").alias("x")],
            [nw.col("x").mean().alias("y")],
            {"x": [-1, 1, 2], "y": [4.0, 0.5, 2.5]},
            ["x"],
        ),
        (
            [nw.col("a")],
            [nw.col("a").count().alias("foo-bar"), nw.all().sum()],
            {"a": [-1, 1, 2], "foo-bar": [1, 2, 2], "x": [4, 1, 5], "y": [1.5, 0, 0]},
            ["a"],
        ),
        (
            [nw.col("a", "y").abs()],
            [nw.col("x").sum()],
            {"a": [1, 1, 2], "y": [0.5, 1.5, 1], "x": [1, 4, 5]},
            ["a", "y"],
        ),
        (
            [nw.col("a").abs().alias("y")],
            [nw.all().sum().name.suffix("c")],
            {"y": [1, 2], "ac": [1, 4], "xc": [5, 5]},
            ["y"],
        ),
        (
            [nw.selectors.by_dtype(nw.Float64()).abs()],
            [nw.selectors.numeric().sum()],
            {"y": [0.5, 1.0, 1.5], "a": [2, 4, -1], "x": [1, 5, 4]},
            ["y"],
        ),
    ],
)
def test_group_by_expr(
    constructor: Constructor,
    keys: list[nw.Expr],
    aggs: list[nw.Expr],
    expected: dict[str, list[Any]],
    sort_by: list[str],
) -> None:
    data = {"a": [1, 1, 2, 2, -1], "x": [0, 1, 2, 3, 4], "y": [0.5, -0.5, 1.0, -1.0, 1.5]}
    df = nw.from_native(constructor(data))
    result = df.group_by(*keys).agg(*aggs).sort(*sort_by)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("keys", "lazy_context"),
    [
        ([nw.col("a").drop_nulls()], pytest.raises(InvalidOperationError)),  # Filtration
        (
            [nw.col("a").alias("foo"), nw.col("a").drop_nulls()],
            pytest.raises(InvalidOperationError),
        ),  # Transform and Filtration
        (
            [nw.col("a").alias("foo"), nw.col("a").max()],
            pytest.raises(ComputeError),
        ),  # Transform and Aggregation
        (
            [nw.col("a").alias("foo"), nw.col("a").cum_max()],
            pytest.raises(InvalidOperationError),
        ),  # Transform and Window
        ([nw.lit(42)], pytest.raises(ComputeError)),  # Literal
        ([nw.lit(42).abs()], pytest.raises(ComputeError)),  # Literal
    ],
)
def test_group_by_raise_if_not_elementwise(
    constructor: Constructor, keys: list[nw.Expr], lazy_context: Any
) -> None:
    data = {"a": [1, 2, 2, None], "b": [0, 1, 2, 3], "x": [1, 2, 3, 4]}
    df = nw.from_native(constructor(data))

    context: Any = (
        lazy_context if isinstance(df, nw.LazyFrame) else pytest.raises(ComputeError)
    )
    with context:
        df.group_by(keys).agg(nw.col("x").max())


@pytest.mark.parametrize(
    "keys", [[nw.col("a").abs()], ["a", nw.col("a").abs().alias("a_test")]]
)
def test_group_by_raise_drop_null_keys_with_exprs(
    constructor: Constructor, keys: list[nw.Expr | str]
) -> None:
    data = {"a": [1, 1, 2, 2, -1], "x": [0, 1, 2, 3, 4], "y": [0.5, -0.5, 1.0, -1.0, 1.5]}
    df = nw.from_native(constructor(data))
    with pytest.raises(
        NotImplementedError, match="drop_null_keys cannot be True when keys contains Expr"
    ):
        df.group_by(*keys, drop_null_keys=True)  # type: ignore[call-overload]


def test_group_by_selector(constructor: Constructor) -> None:
    data = {"a": [1, 1, 1], "b": [4, 4, 6], "c": [7.5, 8.5, 9.0]}
    result = (
        nw.from_native(constructor(data))
        .group_by(nw.selectors.by_dtype(nw.Int64))
        .agg(nw.col("c").mean())
        .sort("a", "b")
    )
    expected = {"a": [1, 1], "b": [4, 6], "c": [8.0, 9.0]}
    assert_equal_data(result, expected)


def test_renaming_edge_case(constructor: Constructor) -> None:
    data = {"a": [0, 0, 0], "_a_tmp": [1, 2, 3], "b": [4, 5, 6]}
    result = nw.from_native(constructor(data)).group_by(nw.col("a")).agg(nw.all().min())
    expected = {"a": [0], "_a_tmp": [1], "b": [4]}
    assert_equal_data(result, expected)


def test_group_by_len_1_column(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    """Based on a failure from marimo.

    - https://github.com/marimo-team/marimo/blob/036fd3ff89ef3a0e598bebb166637028024f98bc/tests/_plugins/ui/_impl/tables/test_narwhals.py#L1098-L1108
    - https://github.com/marimo-team/marimo/blob/036fd3ff89ef3a0e598bebb166637028024f98bc/marimo/_plugins/ui/_impl/tables/narwhals_table.py#L163-L188
    """
    if any(x in str(constructor) for x in ("dask",)):
        # `dask`
        #     ValueError: conflicting aggregation functions: [('size', 'a'), ('size', 'a')]
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 2, 1, 2, 3, 4]}
    expected = {"a": [1, 2, 3, 4], "len": [2, 2, 1, 1], "len_a": [2, 2, 1, 1]}
    result = (
        nw.from_native(constructor(data))
        .group_by("a")
        .agg(nw.len(), nw.len().alias("len_a"))
        .sort("a")
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        ("A", "B"),
        (1.5, 5.2),
        (dt.datetime(2000, 1, 1), dt.datetime(2002, 1, 1)),
        (dt.date(2000, 1, 1), dt.date(2002, 1, 1)),
        (dt.time(5, 0, 0), dt.time(14, 0, 0)),
        (dt.timedelta(32), dt.timedelta(800)),
        (False, True),
        (b"a", b"z"),
        (Decimal("43.954"), Decimal("264.124")),
    ],
    ids=[
        "str",
        "float",
        "datetime",
        "date",
        "time",
        "timedelta",
        "bool",
        "bytes",
        "Decimal",
    ],
)
def test_group_by_no_preserve_dtype(
    constructor_eager: ConstructorEager, low: NonNestedLiteral, high: NonNestedLiteral
) -> None:
    """Minimal repro for [`px.sunburst` failure].

    The issue appeared for `n_unique`, but applies for any [aggregation that requires a function].

    [`px.sunburst` failure]: https://github.com/narwhals-dev/narwhals/pull/2680#discussion_r2151972940
    [aggregation that requires a function]: https://github.com/pandas-dev/pandas/issues/57317
    """
    if (
        "polars" in str(constructor_eager)
        and isinstance(low, Decimal)
        and POLARS_VERSION < (1, 21, 0)
    ):
        pytest.skip("Decimal support in group_by for polars didn't stabilize until 1.0.0")
    data = {
        "col_a": ["A", "B", None, "A", "A", "B", None],
        "col_b": [low, low, high, high, None, None, None],
    }
    expected = {"col_a": [None, "A", "B"], "n_unique": [2, 3, 2]}
    frame = nw.from_native(constructor_eager(data))
    result = (
        frame.group_by("col_a").agg(n_unique=nw.col("col_b").n_unique()).sort("col_a")
    )
    actual_dtype = result.schema["n_unique"]
    assert actual_dtype.is_integer()
    assert_equal_data(result, expected)
