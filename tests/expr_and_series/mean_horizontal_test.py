from __future__ import annotations

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
import narwhals.selectors as ncs
from tests.conftest import dask_pyarrow_constructor
from tests.utils import (
    PANDAS_VERSION,
    Constructor,
    assert_equal_data,
    xfail_if_builtin_dask,
)


def test_meanh(constructor: Constructor) -> None:
    data = {"a": [1, 3, None, None], "b": [4, None, 6, None]}
    df = nw.from_native(constructor(data))
    mean = nw.mean_horizontal(nw.col("a"), nw.col("b"))
    # `assert_equal_data` treats NaN and None as equal, so check `is_null` explicitly
    result = df.select(horizontal_mean=mean, is_null=mean.is_null())
    expected = {
        "horizontal_mean": [2.5, 3.0, 6.0, None],
        "is_null": [False, False, False, True],
    }
    assert_equal_data(result, expected)


def test_meanh_with_literal(constructor: Constructor) -> None:
    data = {"a": [1, 3, None, None], "b": [4, None, 6, None]}
    df = nw.from_native(constructor(data))
    result = df.select(horizontal_mean=nw.mean_horizontal(nw.lit(1), "a", nw.col("b")))
    expected = {"horizontal_mean": [2.0, 2.0, 3.5, 1.0]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("other", "expected"), [(nw.sum("b"), 10.5), (nw.lit(2, nw.Int64), 4.0)]
)
def test_mean_horizontal_of_scalars(
    arrow_constructor: Constructor, other: nw.Expr, expected: float
) -> None:
    df = nw.from_native(arrow_constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = df.select(mean=nw.mean_horizontal(nw.sum("a"), other))
    assert_equal_data(result, {"mean": [expected]})


@pytest.mark.parametrize("expr", [ncs.boolean(), ncs.boolean().sum()])
def test_dask_mean_horizontal_empty_selector(expr: nw.Expr) -> None:
    df = nw.from_native(dask_pyarrow_constructor({"a": [1, 2]}))
    with pytest.raises(
        (TypeError, ValueError), match=r"empty iterable|at least one input column"
    ):
        df.select(nw.mean_horizontal(expr)).collect()


def test_arrow_horizontal_mean_nulls(
    arrow_constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    df = nw.from_native(arrow_constructor({"a": [None, 3], "b": [None, 7]}))
    if PANDAS_VERSION < PkgVersion("3"):
        xfail_if_builtin_dask(df, request)
    mean = nw.mean_horizontal("a", "b")
    assert_equal_data(
        df.select(mean=mean, null=mean.is_null()),
        {"mean": [None, 5.0], "null": [True, False]},
    )


def test_mean_horizontal_with_grouped_expression(
    arrow_constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    df = nw.from_native(arrow_constructor({"a": [1, 2, 3, 4], "g": ["x", "y"] * 2}))
    xfail_if_builtin_dask(df, request)
    result = df.select("a", mean=nw.mean_horizontal("a", nw.col("a").sum().over("g")))
    assert_equal_data(result.sort("a"), {"a": [1, 2, 3, 4], "mean": [2.5, 4, 3.5, 5]})


def test_dask_horizontal_mean_with_typed_literal() -> None:
    dd = pytest.importorskip("dask.dataframe")
    pytest.importorskip("pyarrow")
    import pandas as pd

    frame = pd.DataFrame(
        {"a": pd.array([1, 2, 3], dtype="int64[pyarrow]")},
        index=pd.Index([9, 4, 9], name="idx"),
    )
    df = nw.from_native(dd.from_pandas(frame, npartitions=2, sort=False))
    result = (
        df.select(mean=nw.mean_horizontal("a", nw.lit(2, nw.Int64))).to_native().compute()
    )
    assert result.index.equals(frame.index)
    assert result["mean"].tolist() == [1.5, 2.0, 2.5]


def test_dask_grouped_mean_with_typed_literal_and_named_index(
    request: pytest.FixtureRequest,
) -> None:
    dd = pytest.importorskip("dask.dataframe")
    pytest.importorskip("pyarrow")
    import pandas as pd

    frame = pd.DataFrame(
        {"a": pd.array([1, 2, 3, 4], dtype="int64[pyarrow]"), "g": ["x", "y"] * 2},
        index=pd.Index([3, 1, 2, 0], name="idx"),
    )
    df = nw.from_native(dd.from_pandas(frame, npartitions=2, sort=False))
    xfail_if_builtin_dask(df, request)
    mean = nw.mean_horizontal(nw.lit(2, nw.Int64), nw.sum("a").over("g"))
    assert_equal_data(
        df.select("a", mean=mean).sort("a"), {"a": [1, 2, 3, 4], "mean": [3, 4, 3, 4]}
    )


def test_meanh_all(constructor: Constructor) -> None:
    data = {"a": [2, 4, 6], "b": [10, 20, 30]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.mean_horizontal(nw.all()))
    expected = {"a": [6, 12, 18]}
    assert_equal_data(result, expected)
    result = df.select(c=nw.mean_horizontal(nw.all()))
    expected = {"c": [6, 12, 18]}
    assert_equal_data(result, expected)
