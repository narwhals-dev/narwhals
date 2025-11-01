from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import IntoSchema
    from tests.conftest import Data


@pytest.fixture(scope="module")
def testing_schema() -> IntoSchema:
    return {
        "int": nw.Int32(),
        "float": nw.Float32(),
        "str": nw.String(),
        "categorical": nw.Categorical(),
        "enum": nw.Enum(["beluga", "narwhal", "orca"]),
        "bool": nw.Boolean(),
        "datetime": nw.Datetime(),
        "date": nw.Date(),
        "time": nw.Time(),
        "duration": nw.Duration(),
        "binary": nw.Binary(),
        "list": nw.List(nw.Float32()),
        "array": nw.Array(nw.Int32(), shape=2),
        "struct": nw.Struct({"a": nw.Int64(), "b": nw.List(nw.String())}),
    }


@pytest.fixture(scope="module")
def testing_data() -> Data:
    return {
        "int": [1, 2, 3, 4],
        "float": [1.0, float("nan"), float("inf"), None],
        "str": ["beluga", "narwhal", "orca", None],
        "categorical": ["beluga", "narwhal", "beluga", None],
        "enum": ["beluga", "narwhal", "orca", "narwhal"],
        "bool": [True, False, True, None],
        "datetime": [
            datetime(2025, 1, 1, 12),
            datetime(2025, 1, 2, 12),
            datetime(2025, 1, 3, 12),
            None,
        ],
        "date": [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3), None],
        "time": [time(9, 0), time(9, 1, 10), time(9, 2), None],
        "duration": [
            timedelta(seconds=1),
            timedelta(seconds=2),
            timedelta(seconds=3),
            None,
        ],
        "binary": [b"foo", b"bar", b"baz", None],
        "list": [[1.0, float("nan")], [], [None], None],
        "array": [[1, 2], [3, 4], [5, 6], None],
        "struct": [
            {"a": 1, "b": ["narwhal", "beluga"]},
            {"a": 2, "b": ["orca"]},
            {"a": 3, "b": [None]},
            None,
        ],
    }
