from __future__ import annotations

from contextlib import nullcontext
from typing import Any
from typing import Mapping

import pandas as pd
import polars as pl
import pytest

import narwhals.stable.v1 as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import PANDAS_VERSION
from tests.utils import PYARROW_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data: Mapping[str, Any] = {"a": [1, 1, 3], "b": [4, 4, 6], "c": [7.0, 8.0, 9.0]}

df_pandas = pd.DataFrame(data)
df_lazy = pl.LazyFrame(data)


def test_group_by_complex() -> None:
    expected = {"a": [1, 3], "b": [-3.5, -3.0]}

    df = nw.from_native(df_pandas)
    with pytest.warns(UserWarning, match="complex group-by"):
        result_pd = nw.to_native(
            df.group_by("a").agg((nw.col("b") - nw.col("c").mean()).mean()).sort("a")
        )
    assert_equal_data(result_pd, expected)

    lf = nw.from_native(df_lazy).lazy()
    result_pl = lf.group_by("a").agg((nw.col("b") - nw.col("c").mean()).mean()).sort("a")
    assert_equal_data(result_pl, expected)


def test_invalid_group_by_dask() -> None:
    pytest.importorskip("dask")
    import dask.dataframe as dd

    df_dask = dd.from_pandas(df_pandas)

    with pytest.raises(ValueError, match=r"Non-trivial complex aggregation found"):
        nw.from_native(df_dask).group_by("a").agg(nw.col("b").mean().min())


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
    constructor: Constructor,
    attr: str,
    expected: dict[str, list[int | float]],
    request: pytest.FixtureRequest,
) -> None:
    if "pandas_pyarrow" in str(constructor) and attr == "var" and PANDAS_VERSION < (2, 1):
        # Known issue with variance calculation in pandas 2.0.x with pyarrow backend in groupby operations"
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 1, 1, 2], "b": [1, None, 2, 3]}
    expr = getattr(nw.col("b"), attr)()
    result = nw.from_native(constructor(data)).group_by("a").agg(expr).sort("a")
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("attr", "ddof"),
    [
        ("std", 0),
        ("var", 0),
        ("std", 2),
        ("var", 2),
    ],
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
    import pandas as pd

    df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    with pytest.raises(ValueError, match="Expected unique output names"):
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
        df.group_by("a")
        .agg(
            b_min=nw.col("b").min(),
            b_max=nw.col("b").max(),
        )
        .sort("a")
    )
    expected = {
        "a": [1, 2],
        "b_min": [4, 6],
        "b_max": [5, 6],
    }
    assert_equal_data(result, expected)


def test_group_by_simple_unnamed(constructor: Constructor) -> None:
    data = {"a": [1, 1, 2], "b": [4, 5, 6], "c": [7, 2, 1]}
    df = nw.from_native(constructor(data)).lazy()
    result = (
        df.group_by("a")
        .agg(
            nw.col("b").min(),
            nw.col("c").max(),
        )
        .sort("a")
    )
    expected = {
        "a": [1, 2],
        "b": [4, 6],
        "c": [7, 1],
    }
    assert_equal_data(result, expected)


def test_group_by_multiple_keys(constructor: Constructor) -> None:
    data = {"a": [1, 1, 2], "b": [4, 4, 6], "c": [7, 2, 1]}
    df = nw.from_native(constructor(data)).lazy()
    result = (
        df.group_by("a", "b")
        .agg(
            c_min=nw.col("c").min(),
            c_max=nw.col("c").max(),
        )
        .sort("a")
    )
    expected = {
        "a": [1, 2],
        "b": [4, 6],
        "c_min": [2, 1],
        "c_max": [7, 1],
    }
    assert_equal_data(result, expected)


def test_key_with_nulls(
    constructor: Constructor,
    request: pytest.FixtureRequest,
) -> None:
    if "modin" in str(constructor):
        # TODO(unassigned): Modin flaky here?
        request.applymarker(pytest.mark.skip)

    context = (
        pytest.raises(NotImplementedError, match="null values")
        if ("pandas_constructor" in str(constructor) and PANDAS_VERSION < (1, 1, 0))
        else nullcontext()
    )
    data = {"b": [4, 5, None], "a": [1, 2, 3]}
    with context:
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


def test_key_with_nulls_iter(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
) -> None:
    if PANDAS_VERSION < (1, 0) and "pandas_constructor" in str(constructor_eager):
        # Grouping by null values is not supported in pandas < 1.0.0
        request.applymarker(pytest.mark.xfail)
    data = {"b": ["4", "5", None, "7"], "a": [1, 2, 3, 4], "c": ["4", "3", None, None]}
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


def test_group_by_categorical(
    constructor: Constructor,
    request: pytest.FixtureRequest,
) -> None:
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (
        15,
    ):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)

    data = {"g1": ["a", "a", "b", "b"], "g2": ["x", "y", "x", "z"], "x": [1, 2, 3, 4]}
    df = nw.from_native(constructor(data))
    result = (
        df.with_columns(
            g1=nw.col("g1").cast(nw.Categorical()),
            g2=nw.col("g2").cast(nw.Categorical()),
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
    if any(x in str(constructor) for x in ("dask", "modin", "cudf")):
        # bugged in dask https://github.com/dask/dask/issues/11612
        # and modin lol https://github.com/modin-project/modin/issues/7414
        # and cudf https://github.com/rapidsai/cudf/issues/17649
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor) and PANDAS_VERSION < (1,):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 1, 2], "b": [4, 5, 6]}))
    result = df.group_by("a").agg(c=nw.col("b").mean(), d=nw.col("b").mean()).sort("a")
    expected = {"a": [1, 2], "c": [4.5, 6], "d": [4.5, 6]}
    assert_equal_data(result, expected)


def test_all_kind_of_aggs(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("dask", "cudf", "modin")):
        # bugged in dask https://github.com/dask/dask/issues/11612
        # and modin lol https://github.com/modin-project/modin/issues/7414
        # and cudf https://github.com/rapidsai/cudf/issues/17649
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor) and PANDAS_VERSION < (1, 4):
        # Bug in old pandas, can't do DataFrameGroupBy[['b', 'b']]
        request.applymarker(pytest.mark.xfail)
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


def test_group_by_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 1, 3], "b": [4, 5, 6]}))
    with pytest.raises(NotImplementedError, match=r"not \(yet\?\) supported"):
        df.group_by(nw.col("a")).agg(nw.col("b").mean())  # type: ignore[arg-type]


def test_pandas_group_by_index_and_column_overlap() -> None:
    df = pd.DataFrame(
        {"a": [1, 1, 2], "b": [4, 5, 6]}, index=pd.Index([0, 1, 2], name="a")
    )
    result = nw.from_native(df, eager_only=True).group_by("a").agg(nw.col("b").mean())
    expected = {"a": [1, 2], "b": [4.5, 6.0]}
    assert_equal_data(result, expected)

    key, result = next(iter(nw.from_native(df, eager_only=True).group_by("a")))
    assert key == (1,)
    expected_native = pd.DataFrame({"a": [1, 1], "b": [4, 5]})
    pd.testing.assert_frame_equal(result.to_native(), expected_native)
