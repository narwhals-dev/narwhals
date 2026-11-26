from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.dependencies import get_polars


@pytest.mark.skipif(get_polars() is None, reason="requires Polars")
@pytest.mark.parametrize(("method", "sign"), [("cum_min", 1), ("cum_max", -1)])
@pytest.mark.parametrize(
    ("dtype_name", "limit"),
    [("Float32", 3.4028234663852886e38), ("Float64", 1.7976931348623157e308)],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_polars_cumulative_infinity(
    method: str, sign: int, dtype_name: str, limit: float, *, reverse: bool
) -> None:
    import polars as pl

    infinity = sign * float("inf")
    bound = sign * limit
    values = [None, infinity, None, infinity, bound, infinity, float(sign)]
    expected = [None, infinity, None, infinity, bound, bound, float(sign)]
    if reverse:
        values.reverse()
        expected.reverse()

    dtype = getattr(pl, dtype_name)
    series = nw.from_native(pl.Series("a", values, dtype=dtype), series_only=True)
    result = getattr(series, method)(reverse=reverse)
    assert result.to_list() == expected
    assert result.dtype == series.dtype

    native = pl.DataFrame(
        {
            "a": pl.Series(values * 2, dtype=dtype),
            "g": [0] * len(values) + [1] * len(values),
        }
    )
    expr = getattr(nw.col("a"), method)(reverse=reverse).over("g")
    result_frame = nw.from_native(native).select(expr)
    assert result_frame["a"].to_list() == expected * 2
    assert result_frame["a"].dtype == series.dtype
