from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("numpy")
import datetime as dt

import narwhals as nw
from narwhals import _plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Sequence


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
            {
                "literal": [
                    dt.date(2020, 1, 26),
                    dt.date(2020, 2, 20),
                    dt.date(2020, 3, 16),
                    dt.date(2020, 4, 10),
                ]
            },
        ),
        (
            (
                nwp.date_range(
                    dt.date(2021, 1, 30),
                    nwp.lit(18747, nw.Int32).cast(nw.Date),
                    interval="90d",
                    closed="left",
                ).alias("date_range_cast_expr"),
                {"date_range_cast_expr": [dt.date(2021, 1, 30)]},
            )
        ),
    ],
)
def test_date_range(
    expr: nwp.Expr | Sequence[nwp.Expr], expected: dict[str, Any], data: dict[str, Any]
) -> None:
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)


def test_date_range_eager() -> None:
    leap_year = 2024
    series_leap = nwp.date_range(
        dt.date(leap_year, 2, 25), dt.date(leap_year, 3, 25), eager="pyarrow"
    )
    series_regular = nwp.date_range(
        dt.date(leap_year + 1, 2, 25),
        dt.date(leap_year + 1, 3, 25),
        interval=dt.timedelta(days=1),
        eager="pyarrow",
    )
    assert len(series_regular) == 29
    assert len(series_leap) == 30

    expected = [
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

    ser = nwp.date_range(
        dt.date(2000, 1, 1), dt.date(2023, 8, 31), interval="987d", eager="pyarrow"
    )
    result = ser.to_list()
    assert result == expected

    expected = [dt.date(2006, 10, 14), dt.date(2013, 7, 27), dt.date(2020, 5, 9)]
    result = nwp.date_range(
        dt.date(2000, 1, 1),
        dt.date(2023, 8, 31),
        interval="354w",
        closed="right",
        eager="pyarrow",
    ).to_list()
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
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)


def test_int_range_eager() -> None:
    ser = nwp.int_range(50, eager="pyarrow")
    assert isinstance(ser, nwp.Series)
    assert ser.to_list() == list(range(50))
    ser = nwp.int_range(50, eager=nw.Implementation.PYARROW)
    assert ser.to_list() == list(range(50))
