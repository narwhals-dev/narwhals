from __future__ import annotations

from contextlib import nullcontext

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION
from tests.utils import PYARROW_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 1, 3], "b": [4, 4, 6], "c": [7.0, 8, 9]}

df_pandas = pd.DataFrame(data)
df_lazy = pl.LazyFrame(data)


def test_group_by_complex() -> None:
    expected = {"a": [1, 3], "b": [-3.5, -3.0]}

    df = nw.from_native(df_pandas)
    with pytest.warns(UserWarning, match="complex group-by"):
        result = nw.to_native(
            df.group_by("a").agg((nw.col("b") - nw.col("c").mean()).mean()).sort("a")
        )
    assert_equal_data(result, expected)

    lf = nw.from_native(df_lazy).lazy()
    result = nw.to_native(
        lf.group_by("a").agg((nw.col("b") - nw.col("c").mean()).mean()).sort("a")
    )
    assert_equal_data(result, expected)


def test_invalid_group_by_dask() -> None:
    pytest.importorskip("dask")
    pytest.importorskip("dask_expr", exc_type=ImportError)
    import dask.dataframe as dd

    df_dask = dd.from_pandas(df_pandas)

    with pytest.raises(ValueError, match=r"Non-trivial complex aggregation found"):
        nw.from_native(df_dask).group_by("a").agg(nw.col("b").mean().min())

    with pytest.raises(ValueError, match="Non-trivial complex aggregation"):
        nw.from_native(df_dask).group_by("a").agg(nw.col("b"))

    with pytest.raises(
        ValueError, match=r"Anonymous expressions are not supported in group_by\.agg"
    ):
        nw.from_native(df_dask).group_by("a").agg(nw.all().mean())


@pytest.mark.filterwarnings("ignore:Found complex group-by expression:UserWarning")
def test_invalid_group_by() -> None:
    df = nw.from_native(df_pandas)
    with pytest.raises(ValueError, match="does your"):
        df.group_by("a").agg(nw.col("b"))
    with pytest.raises(
        ValueError, match=r"Anonymous expressions are not supported in group_by\.agg"
    ):
        df.group_by("a").agg(nw.all().mean())
    with pytest.raises(
        ValueError, match=r"Anonymous expressions are not supported in group_by\.agg"
    ):
        nw.from_native(pa.table({"a": [1, 2, 3]})).group_by("a").agg(nw.all().mean())
    with pytest.raises(ValueError, match=r"Non-trivial complex aggregation found"):
        nw.from_native(pa.table({"a": [1, 2, 3]})).group_by("a").agg(
            nw.col("b").mean().min()
        )


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


@pytest.mark.parametrize(
    ("attr", "expected"),
    [
        ("sum", {"a": [1, 2], "b": [3, 3]}),
        ("mean", {"a": [1, 2], "b": [1.5, 3]}),
        ("max", {"a": [1, 2], "b": [2, 3]}),
        ("min", {"a": [1, 2], "b": [1, 3]}),
        ("std", {"a": [1, 2], "b": [0.707107, None]}),
        ("len", {"a": [1, 2], "b": [3, 1]}),
        ("n_unique", {"a": [1, 2], "b": [3, 1]}),
        ("count", {"a": [1, 2], "b": [2, 1]}),
    ],
)
def test_group_by_depth_1_agg(
    constructor: Constructor,
    attr: nw.Expr,
    expected: dict[str, list[int | float]],
    request: pytest.FixtureRequest,
) -> None:
    if "cudf" in str(constructor) and attr == "n_unique":
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 1, 1, 2], "b": [1, None, 2, 3]}
    expr = getattr(nw.col("b"), attr)()  # type: ignore[call-overload]
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


def test_group_by_n_unique_w_missing(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor):
        # Issue in cuDF https://github.com/rapidsai/cudf/issues/16861
        request.applymarker(pytest.mark.xfail)

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
    with pytest.raises(ValueError, match="two aggregations with the same"):
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
        .collect()
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
        .collect()
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
        .collect()
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
        expected = {"b": [4.0, 5, float("nan")], "len": [1, 1, 1], "a": [1, 2, 3]}
        assert_equal_data(result, expected)


def test_key_with_nulls_ignored(
    constructor: Constructor,
) -> None:
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
    if PANDAS_VERSION < (1, 3) and "pandas_constructor" in str(constructor_eager):
        # bug in old pandas
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
    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (
        15,
        0,
        0,
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


@pytest.mark.filterwarnings("ignore:Found complex group-by expression:UserWarning")
def test_group_by_shift_raises(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor):
        # Polars supports all kinds of crazy group-by aggregations, so
        # we don't check that it errors here.
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor):
        # This operation fails completely in cuDF anyway, we just let raise its own
        # error.
        request.applymarker(pytest.mark.xfail)
    df_native = {"a": [1, 2, 3], "b": [1, 1, 2]}
    df = nw.from_native(constructor(df_native))
    with pytest.raises(
        ValueError, match=".*(failed to aggregate|Non-trivial complex aggregation found)"
    ):
        df.group_by("b").agg(nw.col("a").shift(1))
