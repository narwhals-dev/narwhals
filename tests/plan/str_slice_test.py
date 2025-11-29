from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.mark.parametrize(
    ("offset", "length", "expected"),
    [(1, 2, {"a": ["da", "df"]}), (-2, None, {"a": ["as", "as"]})],
)
def test_str_slice(offset: int, length: int | None, expected: Data) -> None:
    data = {"a": ["fdas", "edfas"]}
    result = dataframe(data).select(nwp.col("a").str.slice(offset, length))
    assert_equal_data(result, expected)
