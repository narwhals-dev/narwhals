from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import POLARS_VERSION, assert_equal_data, maybe_collect

if TYPE_CHECKING:
    from narwhals.typing import IntoDType, NonNestedLiteral
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[2, 2, 3, None, None], None, []]}
expected = {"a": [True, None, False]}

INT_LIST_DATA = {"a": [[2, 2, 3, None, None], None, [], [None], [1], [0, -1]]}

UNSUPPORTED_EAGER_BACKENDS = ("cudf", "modin", "pandas", "pyarrow")
"""Eager backends where `list.contains` is not implemented."""

UNSUPPORTED_BACKENDS = ("dask", *UNSUPPORTED_EAGER_BACKENDS)
"""Backends where `list.contains` is not implemented."""

SQL_BACKENDS = ("duckdb", "ibis", "pyspark", "sqlframe")
"""Backends which coerce the item to the inner dtype instead of matching polars."""


def xfail_unsupported(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(backend in str(constructor) for backend in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail(reason="`list.contains` unsupported"))


def test_contains_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    xfail_unsupported(request, constructor)
    result = nw.from_native(constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.contains(2)
    )
    assert_equal_data(result, expected)


def test_contains_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(backend in str(constructor_eager) for backend in UNSUPPORTED_EAGER_BACKENDS):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.contains(2)
    assert_equal_data({"a": result}, expected)


@pytest.mark.parametrize(
    ("data", "inner", "item", "expected"),
    [
        pytest.param(
            INT_LIST_DATA,
            nw.Int64(),
            2.0,
            [True, None, False, False, False, False],
            id="float_matches_int",
        ),
        pytest.param(
            INT_LIST_DATA,
            nw.Int64(),
            1.5,
            [False, None, False, False, False, False],
            id="non_integer",
        ),
        pytest.param(
            INT_LIST_DATA,
            nw.Int8(),
            300,
            [False, None, False, False, False, False],
            id="overflow",
        ),
        pytest.param(
            {"a": [[1.0, 2.0], [3.0], None, []]},
            nw.Float64(),
            1,
            [True, False, None, False],
            id="int_matches_float",
        ),
    ],
)
def test_contains_numeric_coercion_expr(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    data: dict[str, Any],
    inner: IntoDType,
    item: float,
    expected: list[bool | None],
) -> None:
    # Mixing numeric kinds is deliberately unspecified, see
    # https://github.com/narwhals-dev/narwhals/issues/3900. This pins what each backend
    # does today so a future change is visible, it is not a guarantee.
    # Polars 2.0 requires an explicit cast rather than lossily coercing to `Float64`;
    # `overflow` compares two integer types, so it still resolves.
    if (
        "polars" in str(constructor)
        and POLARS_VERSION >= (2,)
        and "overflow" not in request.node.callspec.id
    ):
        request.applymarker(pytest.mark.xfail(reason="polars 2.0 needs a cast"))
    xfail_unsupported(request, constructor)
    result = nw.from_native(constructor(data)).select(
        nw.col("a").cast(nw.List(inner)).list.contains(item)
    )
    assert_equal_data(result, {"a": expected})


def test_contains_none_item_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 24, 0):
        request.applymarker(pytest.mark.xfail(reason="old polars null item"))
    if any(backend in str(constructor) for backend in SQL_BACKENDS):
        request.applymarker(pytest.mark.xfail(reason="null item returns null"))
    xfail_unsupported(request, constructor)
    result = nw.from_native(constructor({"a": [[2, None], [1], None, []]})).select(
        nw.col("a").cast(nw.List(nw.Int64())).list.contains(None)
    )
    assert_equal_data(result, {"a": [True, False, None, False]})


@pytest.mark.parametrize(
    ("data", "inner", "item"),
    [
        pytest.param({"a": [[2, 3], [1], None]}, nw.Int64(), True, id="bool_item"),
        pytest.param({"a": [[2, 3], [1], None]}, nw.Int64(), "2", id="str_item"),
        pytest.param(
            {"a": [[datetime(2020, 1, 1, 1, 2, 3)], [], None]},
            nw.Datetime("ns"),
            datetime(2020, 1, 1, 1, 2, 3),
            id="datetime_precision",
        ),
    ],
)
def test_contains_invalid_item_raises(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    data: dict[str, Any],
    inner: IntoDType,
    item: NonNestedLiteral,
) -> None:
    if (
        "datetime_precision" in request.node.callspec.id
        and "polars" in str(constructor)
        and POLARS_VERSION < (1, 28, 0)
    ):
        request.applymarker(pytest.mark.xfail(reason="old polars coerces precision"))
    if any(backend in str(constructor) for backend in SQL_BACKENDS):
        request.applymarker(pytest.mark.xfail(reason="mismatched item coerced"))
    xfail_unsupported(request, constructor)
    df = nw.from_native(constructor(data))
    with pytest.raises(InvalidOperationError):
        maybe_collect(df.select(nw.col("a").cast(nw.List(inner)).list.contains(item)))


def test_contains_all_null_inner_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(backend in str(constructor) for backend in ("ibis", "pyspark")):
        pytest.skip(reason="cannot infer the type of an all-null column")
    xfail_unsupported(request, constructor)
    result = nw.from_native(constructor({"a": [[None, None]]})).select(
        nw.col("a").cast(nw.List(nw.Int64())).list.contains(1)
    )
    assert_equal_data(result, {"a": [False]})
