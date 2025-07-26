from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals import Implementation
from tests.utils import assert_equal_data
from narwhals.exceptions import ComputeError

if TYPE_CHECKING:
    from narwhals.dtypes import DType, IntegerType

EAGER_BACKENDS = (
    Implementation.PANDAS,
    Implementation.PYARROW,
    Implementation.POLARS,
)

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
    series = nw.int_range(start=start, end=end, step=step, dtype=dtype, eager=impl)

    assert series.dtype == dtype
    if end is None:
        end = start
        start = 0
    assert_equal_data({"a": series}, {"a": list(range(start, end, step))})


@pytest.mark.parametrize("dtype", [nw.List, nw.Float64(), nw.Float32, nw.Decimal, nw.String()])
def test_int_range_non_int_dtype(dtype: DType) -> None:

    msg = f"non-integer `dtype` passed to `int_range`: {dtype}"
    with pytest.raises(ComputeError, match=msg):
        nw.int_range(start=0, end=3, dtype=dtype, eager=None)  # type: ignore[arg-type]
