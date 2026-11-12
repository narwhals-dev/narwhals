from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import DUCKDB_VERSION, PANDAS_VERSION, assert_equal_data

if TYPE_CHECKING:
    from typing import Any

    from tests.utils import Constructor, ConstructorEager

data = {"a": [[2, 2, 3, None, None], None, [], [None]]}
expected = {2, 3, None}

MAINTAIN_ORDER_UNSUPPORTED = ("cudf", "dask", "duckdb", "ibis", "modin", "sqlframe")


def _skip_old_pandas(constructor: Constructor | ConstructorEager) -> None:
    if "pandas" in str(constructor):
        if PANDAS_VERSION < PkgVersion("2.2"):
            pytest.skip()
        pytest.importorskip("pyarrow")


def test_unique_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(backend in str(constructor) for backend in ("dask", "modin", "cudf")):
        request.applymarker(pytest.mark.xfail)
    _skip_old_pandas(constructor)
    if "duckdb" in str(constructor) and DUCKDB_VERSION < PkgVersion("1.3"):
        pytest.skip()

    result = (
        nw.from_native(constructor(data))
        .select(nw.col("a").cast(nw.List(nw.Int32())).list.unique())
        .lazy()
        .collect()["a"]
        .to_list()
    )
    # We don't yet have `.list.sort` to get deterministic order, and pyarrow
    # doesn't support `explode`, so we can't guarantee the order of elements.
    # However, we can check that the unique values are present.
    assert len(result) == 4
    assert len(result[0]) == 3
    assert set(result[0]) == {2, 3, None}
    assert result[1] is None or str(result[1]) == "<NA>"
    assert len(result[2]) == 0
    assert len(result[3]) == 1


def test_unique_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(backend in str(constructor_eager) for backend in ("modin", "cudf")):
        request.applymarker(pytest.mark.xfail)
    _skip_old_pandas(constructor_eager)
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.unique().to_list()
    assert len(result) == 4
    assert len(result[0]) == 3
    assert set(result[0]) == {2, 3, None}
    assert result[1] is None or str(result[1]) == "<NA>"
    assert len(result[2]) == 0
    assert len(result[3]) == 1


maintain_order_data = {"a": [[1, 1, 2, 3, 2, None, None], [3, 2, 1], None, []]}
maintain_order_expected = [[1, 2, 3, None], [3, 2, 1], None, []]


def test_unique_maintain_order_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(backend in str(constructor) for backend in MAINTAIN_ORDER_UNSUPPORTED):
        request.applymarker(pytest.mark.xfail)
    _skip_old_pandas(constructor)
    if "duckdb" in str(constructor) and DUCKDB_VERSION < PkgVersion("1.3"):
        pytest.skip()
    result = nw.from_native(constructor(maintain_order_data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.unique(maintain_order=True)
    )
    assert_equal_data(result, {"a": maintain_order_expected})


def test_unique_maintain_order_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(backend in str(constructor_eager) for backend in MAINTAIN_ORDER_UNSUPPORTED):
        request.applymarker(pytest.mark.xfail)
    _skip_old_pandas(constructor_eager)
    df = nw.from_native(constructor_eager(maintain_order_data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.unique(maintain_order=True)
    assert_equal_data({"a": result}, {"a": maintain_order_expected})


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        (
            [["b", "a", "b", None, "a", None], None, [], [None]],
            nw.String(),
            [["b", "a", None], None, [], [None]],
        ),
        (
            [[1.5, float("nan"), 1.5, float("nan")], [-0.0, 2.0, 0.0]],
            nw.Float64(),
            [[1.5, float("nan")], [-0.0, 2.0]],
        ),
    ],
    ids=["string", "float_nan"],
)
def test_unique_maintain_order_series_dtypes(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    values: list[Any],
    dtype: nw.dtypes.DType,
    expected: list[Any],
) -> None:
    if any(backend in str(constructor_eager) for backend in ("modin", "cudf")):
        request.applymarker(pytest.mark.xfail)
    _skip_old_pandas(constructor_eager)
    df = nw.from_native(constructor_eager({"a": values}), eager_only=True)
    result = df["a"].cast(nw.List(dtype)).list.unique(maintain_order=True)
    if dtype == nw.Float64():
        # `assert_equal_data` can't compare NaN nested in lists. NaN is kept once, as in Polars.
        assert str(result.to_list()) == str(expected)
    else:
        assert_equal_data({"a": result}, {"a": expected})


def test_unique_pyarrow_chunked_and_large_list() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    native = pa.chunked_array(
        [
            pa.array([[3, 3, 1], None], type=pa.large_list(pa.int64())),
            pa.array([], type=pa.large_list(pa.int64())),
            pa.array([[None, 2, None, 2]], type=pa.large_list(pa.int64())),
        ]
    )
    result = nw.from_native(native, series_only=True).list.unique(maintain_order=True)
    assert result.to_native().type == pa.large_list(pa.int64())
    assert result.to_list() == [[3, 1], None, [None, 2]]
