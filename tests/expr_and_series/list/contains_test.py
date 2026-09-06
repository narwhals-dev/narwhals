from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

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


def test_contains_numeric_coercion_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    # Different numeric widths/families coerce without matching spuriously:
    # `2.0` matches ints, `1.5` does not, `300` overflows `Int8`, and `1`
    # matches a `Float64` list.
    if any(
        backend in str(constructor)
        for backend in ("dask", "modin", "cudf", "pyarrow", "pandas")
    ):
        request.applymarker(pytest.mark.xfail)
    data_num = {"a": [[2, 2, 3, None, None], None, [], [None], [1], [0, -1]]}
    cases: list[tuple[nw.DType, int | float, list[bool | None]]] = [
        (nw.Int64(), 2.0, [True, None, False, False, False, False]),
        (nw.Int64(), 1.5, [False, None, False, False, False, False]),
        (nw.Int8(), 300, [False, None, False, False, False, False]),
    ]
    for inner, item, expected_num in cases:
        result = nw.from_native(constructor(data_num)).select(
            nw.col("a").cast(nw.List(inner)).list.contains(item)
        )
        assert_equal_data(result, {"a": expected_num})
    result = nw.from_native(constructor({"a": [[1.0, 2.0], [3.0], None, []]})).select(
        nw.col("a").cast(nw.List(nw.Float64())).list.contains(1)
    )
    assert_equal_data(result, {"a": [True, False, None, False]})


def test_contains_all_null_inner_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    # A list of only nulls contains nothing.
    if "ibis" in str(constructor):
        pytest.skip(reason="ibis cannot create all-null column")
    if any(
        backend in str(constructor)
        for backend in ("dask", "modin", "cudf", "pyarrow", "pandas")
    ):
        request.applymarker(pytest.mark.xfail)
    result = nw.from_native(constructor({"a": [[None, None]]})).select(
        nw.col("a").cast(nw.List(nw.Int64())).list.contains(1)
    )
    assert_equal_data(result, {"a": [False]})
