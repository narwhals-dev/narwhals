from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

import pytest

from narwhals.exceptions import ShapeError
from tests.utils import PYARROW_VERSION

if PYARROW_VERSION < (21,):  # pragma: no cover
    pytest.importorskip("numpy")
import datetime as dt

import narwhals as nw
from narwhals import _plan as nwp
from tests.plan.utils import assert_equal_data, assert_equal_series, dataframe

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals.typing import ClosedInterval, EagerAllowed, IntoDType


@pytest.fixture(scope="module")
def data() -> dict[str, Any]:
    """Variant of `compliant_test.data_small`, with only numeric data."""
    return {
        "b": [1, 2, 3],
        "c": [9, 2, 4],
        "d": [8, 7, 8],
        "e": [None, 9, 7],
        "j": [12.1, None, 4.0],
        "k": [42, 10, None],
        "l": [4, 5, 6],
        "m": [0, 1, 2],
    }


@pytest.fixture(scope="module", params=[2024, 2400])
def leap_year(request: pytest.FixtureRequest) -> int:
    result: int = request.param
    return result


EXPECTED_DATE_1: Final = [
    dt.date(2020, 1, 26),
    dt.date(2020, 2, 20),
    dt.date(2020, 3, 16),
    dt.date(2020, 4, 10),
]
EXPECTED_DATE_2: Final = [dt.date(2021, 1, 30)]
EXPECTED_DATE_3: Final = [
    dt.date(2000, 1, 1),
    dt.date(2002, 9, 14),
    dt.date(2005, 5, 28),
    dt.date(2008, 2, 9),
    dt.date(2010, 10, 23),
    dt.date(2013, 7, 6),
    dt.date(2016, 3, 19),
    dt.date(2018, 12, 1),
    dt.date(2021, 8, 14),
]
EXPECTED_DATE_4: Final = [
    dt.date(2006, 10, 14),
    dt.date(2013, 7, 27),
    dt.date(2020, 5, 9),
]


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            [
                nwp.date_range(
                    dt.date(2020, 1, 1),
                    dt.date(2020, 4, 30),
                    interval="25d",
                    closed="none",
                )
            ],
            {"literal": EXPECTED_DATE_1},
        ),
        (
            (
                nwp.date_range(
                    dt.date(2021, 1, 30),
                    nwp.lit(18747, nw.Int32).cast(nw.Date),
                    interval="90d",
                    closed="left",
                ).alias("date_range_cast_expr"),
                {"date_range_cast_expr": EXPECTED_DATE_2},
            )
        ),
    ],
)
def test_date_range(
    expr: nwp.Expr | Sequence[nwp.Expr],
    expected: dict[str, Any],
    data: dict[str, list[dt.date]],
) -> None:
    pytest.importorskip("pyarrow")
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)


def test_date_range_eager_leap(eager: EagerAllowed, leap_year: int) -> None:
    series_leap = nwp.date_range(
        dt.date(leap_year, 2, 25), dt.date(leap_year, 3, 25), eager=eager
    )
    series_regular = nwp.date_range(
        dt.date(leap_year + 1, 2, 25),
        dt.date(leap_year + 1, 3, 25),
        interval=dt.timedelta(days=1),
        eager=eager,
    )
    assert len(series_regular) == 29
    assert len(series_leap) == 30


@pytest.mark.parametrize(
    ("start", "end", "interval", "closed", "expected"),
    [
        (dt.date(2000, 1, 1), dt.date(2023, 8, 31), "987d", "both", EXPECTED_DATE_3),
        (dt.date(2000, 1, 1), dt.date(2023, 8, 31), "354w", "right", EXPECTED_DATE_4),
    ],
)
def test_date_range_eager(
    start: dt.date,
    end: dt.date,
    interval: str | dt.timedelta,
    closed: ClosedInterval,
    expected: list[dt.date],
    eager: EagerAllowed,
) -> None:
    ser = nwp.date_range(start, end, interval=interval, closed=closed, eager=eager)
    result = ser.to_list()
    assert result == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ([nwp.int_range(5)], {"literal": [0, 1, 2, 3, 4]}),
        ([nwp.int_range(nwp.len())], {"literal": [0, 1, 2]}),
        (nwp.int_range(nwp.len() * 5, 20).alias("lol"), {"lol": [15, 16, 17, 18, 19]}),
        (nwp.int_range(nwp.col("b").min() + 4, nwp.col("d").last()), {"b": [5, 6, 7]}),
    ],
)
def test_int_range(
    expr: nwp.Expr | Sequence[nwp.Expr], expected: dict[str, Any], data: dict[str, Any]
) -> None:
    pytest.importorskip("pyarrow")
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)


def test_int_range_eager(eager: EagerAllowed) -> None:
    ser = nwp.int_range(50, eager=eager)
    assert isinstance(ser, nwp.Series)
    assert ser.to_list() == list(range(50))


@pytest.mark.parametrize(("start", "end"), [(0, 0), (0, 1), (-1, 0), (-2.1, 3.4)])
@pytest.mark.parametrize("num_samples", [0, 1, 2, 5, 1_000])
@pytest.mark.parametrize("interval", ["both", "left", "right", "none"])
def test_linear_space_values(
    start: float,
    end: float,
    num_samples: int,
    interval: ClosedInterval,
    *,
    eager_or_false: EagerAllowed | Literal[False],
) -> None:
    # NOTE: Adapted from https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/py-polars/tests/unit/functions/range/test_linear_space.py#L19-L56
    if eager_or_false:
        result = nwp.linear_space(
            start, end, num_samples, closed=interval, eager=eager_or_false
        ).rename("ls")
    else:
        result = (
            dataframe({})
            .select(ls=nwp.linear_space(start, end, num_samples, closed=interval))
            .to_series()
        )

    pytest.importorskip("numpy")
    import numpy as np

    if interval == "both":
        expected = np.linspace(start, end, num_samples)
    elif interval == "left":
        expected = np.linspace(start, end, num_samples, endpoint=False)
    elif interval == "right":
        expected = np.linspace(start, end, num_samples + 1)[1:]
    else:
        expected = np.linspace(start, end, num_samples + 2)[1:-1]

    assert_equal_series(result, expected, "ls")


def test_linear_space_expr() -> None:
    # NOTE: Adapted from https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/py-polars/tests/unit/functions/range/test_linear_space.py#L59-L68
    pytest.importorskip("pyarrow")
    df = dataframe({"a": [1, 2, 3, 4, 5]})

    result = df.select(nwp.linear_space(0, nwp.col("a").len(), 3))
    expected = df.select(
        literal=nwp.Series.from_iterable(
            [0.0, 2.5, 5.0], dtype=nw.Float64, backend="pyarrow"
        )
    )
    assert_equal_data(result, expected)

    result = df.select(nwp.linear_space(nwp.col("a").len(), 0, 3))
    expected = df.select(
        a=nwp.Series.from_iterable([5.0, 2.5, 0.0], dtype=nw.Float64, backend="pyarrow")
    )
    assert_equal_data(result, expected)


# NOTE: More general "supertyping" behavior would need `pyarrow.unify_schemas`
# (https://arrow.apache.org/docs/14.0/python/generated/pyarrow.unify_schemas.html)
@pytest.mark.parametrize(
    ("dtype_start", "dtype_end", "dtype_expected"),
    [
        pytest.param(
            nw.Float32,
            nw.Float32,
            nw.Float32,
            marks=pytest.mark.xfail(
                reason="Didn't preserve `Float32` dtype, promoted to `Float64`",
                raises=AssertionError,
            ),
        ),
        (nw.Float32, nw.Float64, nw.Float64),
        (nw.Float64, nw.Float32, nw.Float64),
        (nw.Float64, nw.Float64, nw.Float64),
        (nw.UInt8, nw.UInt32, nw.Float64),
        (nw.Int16, nw.Int128, nw.Float64),
        (nw.Int8, nw.Float64, nw.Float64),
    ],
)
def test_linear_space_expr_numeric_dtype(
    dtype_start: IntoDType, dtype_end: IntoDType, dtype_expected: IntoDType
) -> None:
    # NOTE: Adapted from https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/py-polars/tests/unit/functions/range/test_linear_space.py#L71-L95
    pytest.importorskip("pyarrow")
    df = dataframe({})
    result = df.select(
        ls=nwp.linear_space(nwp.lit(0, dtype=dtype_start), nwp.lit(1, dtype=dtype_end), 6)
    )
    expected = df.select(
        ls=nwp.Series.from_iterable(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=dtype_expected, backend="pyarrow"
        )
    )
    assert result.get_column("ls").dtype == dtype_expected
    assert_equal_data(result, expected)


def test_linear_space_expr_wrong_length() -> None:
    # NOTE: Adapted from https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/py-polars/tests/unit/functions/range/test_linear_space.py#L194-L199
    pytest.importorskip("pyarrow")
    df = dataframe({"a": [1, 2, 3, 4, 5]})
    with pytest.raises(ShapeError, match="Expected object of length 6, got 5"):
        df.with_columns(nwp.linear_space(0, 1, 6))
