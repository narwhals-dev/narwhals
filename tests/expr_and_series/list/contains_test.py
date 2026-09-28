from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, assert_equal_data, pyspark_session

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[2, 2, 3, None, None], None, []]}
expected = {"a": [True, None, False]}


def test_contains_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(
        backend in str(constructor)
        for backend in ("dask", "modin", "cudf", "pyarrow", "pandas")
    ):
        request.applymarker(pytest.mark.xfail)
    result = nw.from_native(constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.contains(2)
    )
    assert_equal_data(result, expected)


def test_contains_no_match_with_null_elements_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(
        backend in str(constructor)
        for backend in ("dask", "modin", "cudf", "pyarrow", "pandas")
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [[1, None], [None], [2, None], None]}))
    result = df.select(nw.col("a").cast(nw.List(nw.Int32())).list.contains(2))
    assert_equal_data(result, {"a": [False, False, True, None]})


def test_contains_none_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(
        backend in str(constructor)
        for backend in ("dask", "modin", "cudf", "pyarrow", "pandas")
    ):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [[1, None], [None], [1, 2], [1, 1], [2, 1, None, 1], [], None]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").cast(nw.List(nw.Int32())).list.contains(None))
    assert_equal_data(result, {"a": [True, True, False, False, True, False, None]})


def test_contains_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(
        backend in str(constructor_eager)
        for backend in ("modin", "cudf", "pyarrow", "pandas")
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.contains(2)
    assert_equal_data({"a": result}, expected)


@pytest.mark.parametrize(
    ("values", "dtype"),
    [(["x", "y"], nw.List(nw.String())), ([[1], [None]], nw.List(nw.List(nw.Int32())))],
)
def test_contains_none_inner_dtypes_expr(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    values: list[Any],
    dtype: nw.List,
) -> None:
    if any(
        backend in str(constructor)
        for backend in ("dask", "modin", "cudf", "pyarrow", "pandas")
    ):
        request.applymarker(pytest.mark.xfail)
    if (
        "polars" in str(constructor)
        and POLARS_VERSION >= (1, 30)
        and dtype.inner == nw.List
    ):
        # Raises in Polars 1.30-1.x; 2.0 works lazily but panics eagerly.
        pytest.skip(reason="Polars' support for nested inner dtypes varies by version")
    x, y = values
    data = {"a": [[x, None], [None], [x, y], [], None]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").cast(dtype).list.contains(None))
    assert_equal_data(result, {"a": [True, True, False, False, None]})


def test_contains_none_single_empty_list_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(
        backend in str(constructor)
        for backend in ("dask", "modin", "cudf", "pyarrow", "pandas")
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [[1], []]})).filter(nw.col("a").list.len() == 0)
    result = df.select(nw.col("a").cast(nw.List(nw.Int32())).list.contains(None))
    assert_equal_data(result, {"a": [False]})


@pytest.mark.slow
def test_contains_none_non_orderable_inner_type_pyspark() -> None:  # pragma: no cover
    pytest.importorskip("pyspark")
    session = pyspark_session()
    native = session.sql(
        "SELECT array(map('k', 1), NULL) AS a UNION ALL SELECT array(map('k', 1))"
    )
    result = nw.from_native(native).select(nw.col("a").list.contains(None))
    assert_equal_data(result, {"a": [True, False]})
