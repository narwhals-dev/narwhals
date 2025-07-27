from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals import Implementation
from narwhals.exceptions import ComputeError
from tests.utils import Constructor, assert_equal_data

if TYPE_CHECKING:
    from narwhals.dtypes import DType, IntegerType

EAGER_BACKENDS = (Implementation.PANDAS, Implementation.PYARROW, Implementation.POLARS)


@pytest.mark.parametrize("impl", EAGER_BACKENDS)
@pytest.mark.parametrize(
    ("start", "end", "step", "dtype"),
    [
        (0, 0, 1, nw.UInt8()),
        (0, 3, 1, nw.UInt16),
        (-3, 0, -1, nw.Int16()),
        (0, 3, 2, nw.Int64),
        (3, None, 1, nw.UInt32),
        (3, None, 2, nw.Int8()),
    ],
)
def test_int_range_eager(
    start: int,
    end: int | None,
    step: int,
    dtype: type[IntegerType] | IntegerType,
    impl: nw.Implementation,
) -> None:
    pytest.importorskip(impl.value)
    series = nw.int_range(start=start, end=end, step=step, dtype=dtype, eager=impl)

    assert series.dtype == dtype
    if end is None:
        end = start
        start = 0
    assert_equal_data({"a": series}, {"a": list(range(start, end, step))})


@pytest.mark.parametrize(
    ("start", "end", "step", "dtype", "expected"),
    [
        (0, nw.len(), 1, nw.UInt8(), [0, 1, 2]),
        (0, 3, 1, nw.UInt16, [0, 1, 2]),
        (-3, nw.len() - 3, 1, nw.Int16(), [-3, -2, -1]),
        (nw.len(), 0, -1, nw.Int64, [3, 2, 1]),
        (nw.len(), None, 1, nw.UInt32, [0, 1, 2]),
    ],
)
def test_int_range_lazy(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    start: int,
    end: int | None,
    step: int,
    dtype: type[IntegerType] | IntegerType,
    expected: list[int],
) -> None:
    if any(x in str(constructor) for x in ("dask", "duckdb", "ibis", "spark")):
        reason = "not implemented yet"
        request.applymarker(pytest.mark.xfail(reason=reason))

    data = {"a": ["foo", "bar", "baz"]}
    int_range = nw.int_range(start=start, end=end, step=step, dtype=dtype, eager=False)
    result = nw.from_native(constructor(data)).select(int_range)

    output_name = "len" if isinstance(start, nw.Expr) and end is not None else "literal"
    assert_equal_data(result, {output_name: expected})
    assert result.collect_schema()[output_name] == dtype


@pytest.mark.parametrize(
    "dtype", [nw.List, nw.Float64(), nw.Float32, nw.Decimal, nw.String()]
)
def test_int_range_non_int_dtype(dtype: DType) -> None:
    msg = f"non-integer `dtype` passed to `int_range`: {dtype}"
    with pytest.raises(ComputeError, match=msg):
        nw.int_range(start=0, end=3, dtype=dtype)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (nw.col("foo", "bar").sum(), nw.col("foo", "bar").sum()),
        (1, nw.col("foo", "bar").sum()),
    ],
)
def test_int_range_multi_named(start: int | nw.Expr, end: int | nw.Expr | None) -> None:
    prefix = "`start`" if isinstance(start, nw.Expr) else "`end`"
    msg = f"{prefix} must contain exactly one value, got expression returning multiple values"
    with pytest.raises(ComputeError, match=msg):
        nw.int_range(start=start, end=end)


def test_int_range_eager_set_to_lazy_backend() -> None:
    with pytest.raises(ValueError, match="Cannot create a Series from a lazy backend"):
        nw.int_range(123, eager=Implementation.DUCKDB)
