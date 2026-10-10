from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    assert_equal_data,
    maybe_collect,
    pyspark_session,
)

if TYPE_CHECKING:
    from narwhals.dtypes import DType
    from narwhals.typing import NonNestedLiteral
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[2, 2, 3, None, None], None, []]}
expected = {"a": [True, None, False]}

INT_LIST_DATA = {"a": [[2, 2, 3, None, None], None, [], [None], [1], [0, -1]]}

SQL_BACKENDS = ("duckdb", "ibis", "pyspark", "sqlframe")
"""Backends which coerce the item to the inner dtype instead of matching polars."""

PYARROW_COMPUTE_BACKENDS = ("dask", "modin", "pandas", "pyarrow")
"""Backends whose `list.contains` runs on pyarrow, which casts a datetime item to the list's unit."""


def skip_or_xfail_unsupported(
    request: pytest.FixtureRequest, constructor: Constructor | ConstructorEager
) -> None:
    if "cudf" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="`list.contains` unsupported"))
    if any(backend in str(constructor) for backend in ("pandas", "modin", "dask")):
        if PANDAS_VERSION < (2, 2):
            pytest.skip(reason="casting to `List` needs pandas>=2.2")
        pytest.importorskip("pyarrow")


def test_contains_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    skip_or_xfail_unsupported(request, constructor)
    result = nw.from_native(constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.contains(2)
    )
    assert_equal_data(result, expected)


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
    inner: DType,
    item: float,
    expected: list[bool | None],
) -> None:
    # Mixing numeric kinds is deliberately unspecified, see
    # https://github.com/narwhals-dev/narwhals/issues/3900. This pins what each backend
    # does today so a future change is visible, it is not a guarantee.
    # Polars 2.0 requires an explicit cast rather than lossily coercing to `Float64`.
    mixes_kinds = isinstance(item, float) != inner.is_float()
    if "polars" in str(constructor) and POLARS_VERSION >= (2,) and mixes_kinds:
        request.applymarker(pytest.mark.xfail(reason="polars 2.0 needs a cast"))
    skip_or_xfail_unsupported(request, constructor)
    result = nw.from_native(constructor(data)).select(
        nw.col("a").cast(nw.List(inner)).list.contains(item)
    )
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("data", "inner", "item"),
    [
        pytest.param({"a": [[2, 3], [1], None]}, nw.Int64(), True, id="bool_item"),
        pytest.param({"a": [[2, 3], [1], None]}, nw.Int64(), "2", id="str_item"),
        pytest.param({"a": [["x"], [], None]}, nw.String(), float("nan"), id="nan_item"),
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
    inner: DType,
    item: NonNestedLiteral,
) -> None:
    if (
        isinstance(item, (float, datetime))
        and "polars" in str(constructor)
        and POLARS_VERSION < (1, 28, 0)
    ):
        request.applymarker(pytest.mark.xfail(reason="old polars coerces the item"))
    coercing_backends = SQL_BACKENDS
    if isinstance(item, datetime):
        coercing_backends += PYARROW_COMPUTE_BACKENDS
    if any(backend in str(constructor) for backend in coercing_backends):
        request.applymarker(pytest.mark.xfail(reason="mismatched item coerced"))
    skip_or_xfail_unsupported(request, constructor)
    df = nw.from_native(constructor(data))
    with pytest.raises(InvalidOperationError):
        maybe_collect(df.select(nw.col("a").cast(nw.List(inner)).list.contains(item)))


def test_contains_invalid_item_raises_without_rows(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(backend in str(constructor) for backend in SQL_BACKENDS):
        request.applymarker(pytest.mark.xfail(reason="mismatched item coerced"))
    skip_or_xfail_unsupported(request, constructor)
    df = nw.from_native(constructor({"a": [[2, 3]]})).filter(nw.col("a").is_null())
    with pytest.raises(InvalidOperationError):
        maybe_collect(df.select(nw.col("a").cast(nw.List(nw.Int64())).list.contains("2")))


def test_contains_all_null_inner_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(backend in str(constructor) for backend in ("ibis", "pyspark")):
        pytest.skip(reason="cannot infer the type of an all-null column")
    skip_or_xfail_unsupported(request, constructor)
    result = nw.from_native(constructor({"a": [[None, None]]})).select(
        nw.col("a").cast(nw.List(nw.Int64())).list.contains(1)
    )
    assert_equal_data(result, {"a": [False]})


def test_contains_no_match_with_null_elements_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    skip_or_xfail_unsupported(request, constructor)
    df = nw.from_native(constructor({"a": [[1, None], [None], [2, None], None]}))
    result = df.select(nw.col("a").cast(nw.List(nw.Int32())).list.contains(2))
    assert_equal_data(result, {"a": [False, False, True, None]})


def test_contains_none_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    skip_or_xfail_unsupported(request, constructor)
    data = {"a": [[1, None], [None], [1, 2], [1, 1], [2, 1, None, 1], [], None]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").cast(nw.List(nw.Int32())).list.contains(None))
    assert_equal_data(result, {"a": [True, True, False, False, True, False, None]})


def test_contains_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    skip_or_xfail_unsupported(request, constructor_eager)
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
    skip_or_xfail_unsupported(request, constructor)
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


@pytest.mark.parametrize(
    ("x", "y", "dtype"),
    [
        ("x", "y", nw.String()),
        (True, False, nw.Boolean()),
        (date(2020, 1, 1), date(2021, 1, 1), nw.Date()),
        (Decimal("1.50"), Decimal("2.50"), nw.Decimal(38, 2)),
        (1.5, float("nan"), nw.Float64()),
        (float("inf"), float("-inf"), nw.Float64()),
    ],
)
def test_contains_inner_dtypes_expr(
    request: pytest.FixtureRequest, constructor: Constructor, x: Any, y: Any, dtype: DType
) -> None:
    skip_or_xfail_unsupported(request, constructor)
    is_nan = isinstance(y, float) and math.isnan(y)
    if (
        "polars" in str(constructor)
        and POLARS_VERSION < (1, 28)
        and (isinstance(dtype, nw.Decimal) or is_nan)
    ):
        pytest.skip(reason="Polars<1.28 doesn't support decimals, nor match NaN")
    if is_nan:
        if "duckdb" in str(constructor):
            reason = "DuckDB turns a NaN literal into NULL."
            request.applymarker(pytest.mark.xfail(reason=reason))
        if PANDAS_VERSION < (3,) and any(
            backend in str(constructor) for backend in ("pandas", "modin", "dask")
        ):
            pytest.skip(reason="pandas<3 turns NaN into null when casting to `List`")
    if "sqlframe" in str(constructor) and isinstance(x, float) and math.isinf(x):
        # https://github.com/eakmanrq/sqlframe/issues/648
        reason = "SQLFrame writes `inf` into SQL unquoted, which DuckDB can't parse."
        request.applymarker(pytest.mark.xfail(reason=reason))
    data = {"a": [[x, None], [y], [x, y], [], None]}
    list_ = nw.col("a").cast(nw.List(dtype))
    result = nw.from_native(constructor(data)).select(
        x=list_.list.contains(x), y=list_.list.contains(y)
    )
    expected = {
        "x": [True, False, True, False, None],
        "y": [False, True, True, False, None],
    }
    assert_equal_data(result, expected)


def test_contains_chunked(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skip_or_xfail_unsupported(request, constructor_eager)
    if "polars" in str(constructor_eager) and (1, 44) <= POLARS_VERSION < (2,):
        # Fixed in Polars 2.0 (https://github.com/pola-rs/polars/pull/29635). Not
        # strict, as only some CPUs hit the bug: arm64 does, while the x86_64 CI
        # runners don't.
        reason = "Polars 1.44 mixes up chunks that are slices of the same list array."
        request.applymarker(pytest.mark.xfail(reason=reason, strict=False))
    pytest.importorskip("pyarrow")
    # Tiny blocks, so that chunks of 3+ values are sliced and smaller ones combined.
    monkeypatch.setattr("narwhals._arrow.utils._LIST_BLOCK_VALUES", 4)
    monkeypatch.setattr("narwhals._arrow.utils._LIST_MIN_BLOCK_VALUES", 3)
    data = {"a": [[1, 2], [3], None, [], [2, None, 2], [4, 4, 4, 2], [5]]}
    contains_2 = [True, False, None, False, True, True, False]
    slices = [(4, 6), (0, 0), (0, 1), (1, 3), (3, 4), (6, 7), (0, 2), (1, 2)]
    df = nw.from_native(constructor_eager(data), eager_only=True).cast(
        {"a": nw.List(nw.Int64())}
    )
    chunked = nw.concat([df[start:stop] for start, stop in slices])
    result = chunked.select(nw.col("a").list.contains(2))
    expected = [contains_2[i] for start, stop in slices for i in range(start, stop)]
    assert_equal_data(result, {"a": expected})


def test_contains_256_matches(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    skip_or_xfail_unsupported(request, constructor)
    # Counted in uint8, 256 matches would wrap around to 0.
    data = {"a": [[1] * 256, [0] * 256, [0] * 255 + [1]]}
    result = nw.from_native(constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.contains(1)
    )
    assert_equal_data(result, {"a": [True, False, True]})


def test_contains_non_pyarrow_list_dask() -> None:
    pytest.importorskip("dask")
    import dask
    import dask.dataframe as dd
    import pandas as pd

    with dask.config.set({"dataframe.convert-string": False}):
        native = dd.from_pandas(pd.DataFrame({"a": [[1], [2]]}), npartitions=1)
    df = nw.from_native(native)
    with pytest.raises(NotImplementedError, match="pyarrow-backed"):
        df.select(nw.col("a").list.contains(1))


def test_contains_none_single_empty_list_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    skip_or_xfail_unsupported(request, constructor)
    # Polars 1.28-1.29 return `True` for `[]` only when it is the sole row, which
    # the multi-row data of `test_contains_none_expr` does not catch.
    df = nw.from_native(constructor({"i": [0, 1], "a": [[1], []]})).filter(
        nw.col("i") == 1
    )
    result = df.select(nw.col("a").cast(nw.List(nw.Int32())).list.contains(None))
    assert_equal_data(result, {"a": [False]})


@pytest.mark.slow
def test_contains_none_non_orderable_inner_type_pyspark() -> None:  # pragma: no cover
    pytest.importorskip("pyspark")
    session = pyspark_session()
    native = session.sql(
        "SELECT 0 AS i, array(map('k', 1), NULL) AS a"
        " UNION ALL SELECT 1, array(map('k', 1))"
    )
    result = nw.from_native(native).select("i", nw.col("a").list.contains(None)).sort("i")
    assert_equal_data(result, {"i": [0, 1], "a": [True, False]})
