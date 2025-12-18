from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, POLARS_VERSION, assert_equal_data

if TYPE_CHECKING:
    from typing import Any

    from tests.utils import Constructor, ConstructorEager


data = {"a": [[3, 2, 2, 4, -10, None, None], [-1], None, [None, None, None], []]}
expected_desc_nulls_last = [
    [4, 3, 2, 2, -10, None, None],
    [-1],
    None,
    [None, None, None],
    [],
]
expected_desc_nulls_first = [
    [None, None, 4, 3, 2, 2, -10],
    [-1],
    None,
    [None, None, None],
    [],
]
expected_asc_nulls_last = [
    [-10, 2, 2, 3, 4, None, None],
    [-1],
    None,
    [None, None, None],
    [],
]
expected_asc_nulls_first = [
    [None, None, -10, 2, 2, 3, 4],
    [-1],
    None,
    [None, None, None],
    [],
]


def test_sort_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(backend in str(constructor) for backend in ("dask", "cudf")):
        request.applymarker(pytest.mark.xfail)
    if "sqlframe" in str(constructor):
        # https://github.com/eakmanrq/sqlframe/issues/559
        # https://github.com/eakmanrq/sqlframe/issues/560
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 5):
        pytest.skip()
    if "pandas" in str(constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    result = nw.from_native(constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.sort()
    )
    assert_equal_data(result, {"a": expected_asc_nulls_first})


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, expected_desc_nulls_last),
        (True, False, expected_desc_nulls_first),
        (False, True, expected_asc_nulls_last),
        (False, False, expected_asc_nulls_first),
    ],
)
def test_sort_expr_args(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    descending: bool,  # noqa: FBT001
    nulls_last: bool,  # noqa: FBT001
    expected: list[Any],
) -> None:
    if any(backend in str(constructor) for backend in ("dask", "cudf")):
        request.applymarker(pytest.mark.xfail)
    if "ibis" in str(constructor) and descending:
        # https://github.com/ibis-project/ibis/issues/11735
        request.applymarker(pytest.mark.xfail)
    if "sqlframe" in str(constructor) and not nulls_last:
        # https://github.com/eakmanrq/sqlframe/issues/559
        # https://github.com/eakmanrq/sqlframe/issues/560
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 5):
        pytest.skip()
    if "pandas" in str(constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    result = nw.from_native(constructor(data)).select(
        nw.col("a")
        .cast(nw.List(nw.Int32()))
        .list.sort(descending=descending, nulls_last=nulls_last)
    )
    assert_equal_data(result, {"a": expected})


def test_sort_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(backend in str(constructor_eager) for backend in ("dask", "cudf")):
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 5):
        pytest.skip()
    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.sort()
    assert_equal_data({"a": result}, {"a": expected_asc_nulls_first})


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, expected_desc_nulls_last),
        (True, False, expected_desc_nulls_first),
        (False, True, expected_asc_nulls_last),
        (False, False, expected_asc_nulls_first),
    ],
)
def test_sort_series_args(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    descending: bool,  # noqa: FBT001
    nulls_last: bool,  # noqa: FBT001
    expected: list[Any],
) -> None:
    if any(backend in str(constructor_eager) for backend in ("dask", "cudf")):
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 5):
        pytest.skip()
    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = (
        df["a"]
        .cast(nw.List(nw.Int32()))
        .list.sort(descending=descending, nulls_last=nulls_last)
    )
    assert_equal_data({"a": result}, {"a": expected})
