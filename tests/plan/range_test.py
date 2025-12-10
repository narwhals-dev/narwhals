from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import pytest

pytest.importorskip("numpy")
import datetime as dt

import narwhals as nw
from narwhals import _plan as nwp
from tests.conftest import TEST_EAGER_BACKENDS
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals.typing import ClosedInterval, EagerAllowed


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


_HAS_IMPLEMENTATION = frozenset((nw.Implementation.PYARROW, "pyarrow"))
"""Using to filter *the source* of `eager_backend` - which includes `polars` and `pandas` when available.

For now, this lets some tests be written in a backend agnostic way.
"""


@pytest.fixture(
    scope="module", params=_HAS_IMPLEMENTATION.intersection(TEST_EAGER_BACKENDS)
)
def eager(request: pytest.FixtureRequest) -> EagerAllowed:
    result: EagerAllowed = request.param
    return result


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
