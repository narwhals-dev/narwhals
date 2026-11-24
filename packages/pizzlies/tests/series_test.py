from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pizzlies as pz
import pytest
from pizzlies._plugin.utils import _enum_class

from narwhals.exceptions import InvalidOperationError


def _values(series: pz.Series[Any]) -> list[Any]:
    return list(series.to_native())


def test_diff_empty() -> None:
    assert _values(pz.new_series("a", [], pz.Int64()).diff()) == []
    assert _values(pz.new_series("a", [1, 3], pz.Int64()).diff()) == [None, 2]


def test_rolling_nan_does_not_leak() -> None:
    s = pz.new_series("a", [math.nan, 1.0, 2.0], pz.Float64())
    result = _values(s.rolling_sum(2))
    assert result[0] is None
    assert math.isnan(result[1])
    assert result[2] == 3.0
    var = _values(s.rolling_var(2))
    assert math.isnan(var[1])
    assert var[2] == 0.5


def test_rolling_sum_precision() -> None:
    s = pz.new_series("a", [1e16, 1.0], pz.Float64())
    assert _values(s.rolling_sum(1)) == [1e16, 1.0]


def test_rolling_var_stability() -> None:
    s = pz.new_series("a", [1e9, 1e9 + 1], pz.Float64())
    assert _values(s.rolling_var(2)) == [None, 0.5]
    assert _values(s.rolling_std(2)) == [None, math.sqrt(0.5)]


def test_rolling_integers_exact() -> None:
    s = pz.new_series("a", [10**18, 1, None, 2], pz.Int64())
    assert _values(s.rolling_sum(2, min_samples=1)) == [10**18, 10**18 + 1, 1, 2]
    assert _values(s.rolling_var(2, min_samples=1)) == [
        None,
        (10**18 - 1) ** 2 / 2,
        None,
        None,
    ]


def test_rolling_infinities() -> None:
    s = pz.new_series("a", [math.inf, 1.0, -math.inf, 2.0, None, 3.0], pz.Float64())
    assert _values(s.rolling_sum(2)) == [None, math.inf, -math.inf, -math.inf, None, None]
    assert _values(s.rolling_mean(2, min_samples=1))[1:] == [
        math.inf,
        -math.inf,
        -math.inf,
        2.0,
        3.0,
    ]
    var = _values(s.rolling_var(3, min_samples=2))
    assert var[0] is None
    assert all(math.isnan(value) for value in var[1:5])
    assert var[5] == 0.5
    assert math.isnan(_values(s.rolling_sum(3))[2])


def test_rolling_non_finite_with_subnormal() -> None:
    s = pz.new_series("a", [math.inf, 5e-324], pz.Float64())
    assert _values(s.rolling_sum(1)) == [math.inf, 5e-324]
    assert _values(s.rolling_mean(2)) == [None, math.inf]
    assert math.isnan(_values(s.rolling_var(2))[1])


def test_rolling_overflow_saturates() -> None:
    s = pz.new_series("a", [1e308, 1e308, -1e308, -1e308], pz.Float64())
    assert _values(s.rolling_sum(2)) == [None, math.inf, 0.0, -math.inf]
    assert _values(s.rolling_mean(2)) == [None, 1e308, 0.0, -1e308]
    assert _values(s.rolling_var(2))[2] == math.inf


def test_rolling_floats_match_fsum() -> None:
    rng = random.Random(0)  # noqa: S311
    values = [rng.choice([None, rng.uniform(-1e6, 1e6), 1e-300]) for _ in range(200)]
    s = pz.new_series("a", values, pz.Float64())
    window_size = 7
    for result, index in zip(
        _values(s.rolling_sum(window_size, min_samples=1)), range(200), strict=True
    ):
        window = [
            v
            for v in values[max(index - window_size + 1, 0) : index + 1]
            if v is not None
        ]
        assert result == (math.fsum(window) if window else None)


def test_rolling_decimal() -> None:
    s = pz.new_series("a", [Decimal("0.1"), None, Decimal("0.2")], pz.Decimal(10, 2))
    assert _values(s.rolling_sum(3, min_samples=1)) == [
        Decimal("0.1"),
        Decimal("0.1"),
        Decimal("0.3"),
    ]
    assert _values(s.rolling_mean(3, min_samples=2)) == [None, None, Decimal("0.15")]
    assert _values(s.rolling_var(3, min_samples=1)) == [None, None, Decimal("0.005")]


def test_str_to_datetime_offset_aware_to_utc() -> None:
    s = pz.new_series("a", ["2020-01-01T00:00:00+02:00", None], pz.String())
    result = _values(s.str.to_datetime("%Y-%m-%dT%H:%M:%S%z"))
    assert result == [datetime(2019, 12, 31, 22, tzinfo=timezone.utc), None]
    assert result[0].utcoffset() == timedelta(0)
    naive = _values(
        pz.new_series("a", ["2020-01-01"], pz.String()).str.to_datetime("%Y-%m-%d")
    )
    assert naive == [datetime(2020, 1, 1)]


def test_dt_truncate_date_stays_date() -> None:
    s = pz.new_series("a", [date(2020, 5, 17)], pz.Date())
    for every, expected in [("1d", date(2020, 5, 17)), ("2d", date(2020, 5, 16))]:
        result = _values(s.dt.truncate(every))
        assert result == [expected]
        assert type(result[0]) is date


def test_dt_truncate_ambiguous_keeps_fold() -> None:
    pytest.importorskip("zoneinfo")
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/New_York")
    first = datetime(2024, 11, 3, 1, 30, tzinfo=tz)
    second = first.replace(fold=1)
    s = pz.new_series("a", [first, second], pz.Datetime("us", "America/New_York"))
    result = _values(s.dt.truncate("1h"))
    assert [value.utcoffset() for value in result] == [
        first.utcoffset(),
        second.utcoffset(),
    ]
    assert [value.astimezone(timezone.utc) for value in result] == [
        datetime(2024, 11, 3, 5, tzinfo=timezone.utc),
        datetime(2024, 11, 3, 6, tzinfo=timezone.utc),
    ]


def test_list_unique_unhashable() -> None:
    s = pz.new_series("a", [[[1], [1], [2]], None], pz.List(pz.List(pz.Int64())))
    assert _values(s.list.unique(maintain_order=True)) == [[[1], [2]], None]


def test_struct_field_missing_key() -> None:
    s = pz.new_series(
        "a",
        [{"x": 1, "y": 2}, {"x": 3}, None],
        pz.Struct({"x": pz.Int64(), "y": pz.Int64()}),
    )
    assert _values(s.struct.field("y")) == [2, None, None]
    with pytest.raises(InvalidOperationError, match="not found"):
        s.struct.field("z")


def test_enum_class_not_evicted() -> None:
    first = _enum_class(("a", "b"))
    for i in range(100):
        _enum_class((str(i),))
    assert _enum_class(("a", "b")) is first


@pytest.mark.parametrize("normalize", [False, True])
def test_value_counts_name_collision(normalize: bool) -> None:  # noqa: FBT001
    name = "proportion" if normalize else "count"
    s = pz.new_series(name, [7, 7, 8])
    with pytest.raises(pz.exceptions.DuplicateError):
        s.value_counts(normalize=normalize)
    assert s.value_counts(name="n", sort=True).to_native() == {name: [7, 8], "n": [2, 1]}


@pytest.mark.parametrize("values", [[None, "null"], [1, "1"]])
def test_to_dummies_name_collision(values: list[Any]) -> None:
    with pytest.raises(pz.exceptions.DuplicateError):
        pz.new_series("a", values).to_dummies()
