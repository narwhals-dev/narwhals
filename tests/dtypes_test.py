from __future__ import annotations

from datetime import timezone
from types import NoneType
from typing import Literal

import pytest

import narwhals.stable.v1 as nw


@pytest.mark.parametrize("time_unit", ["us", "ns", "ms"])
@pytest.mark.parametrize("time_zone", ["Europe/Rome", timezone.utc, None])
def test_datetime_valid(
    time_unit: Literal["us", "ns", "ms"], time_zone: str | timezone | None
) -> None:
    dtype = nw.Datetime(time_unit=time_unit, time_zone=time_zone)

    assert dtype.time_unit == time_unit
    assert isinstance(dtype.time_zone, (str, NoneType))


@pytest.mark.parametrize("time_unit", ["abc", "s"])
def test_datetime_invalid(time_unit: str) -> None:
    with pytest.raises(ValueError, match="invalid `time_unit`"):
        nw.Datetime(time_unit=time_unit)  # type: ignore[arg-type]
