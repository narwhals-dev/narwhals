from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from narwhals import _plan as nwp
from narwhals._plan import selectors as ncs
from narwhals.exceptions import InvalidOperationError
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from tests.conftest import Data


XFAIL_NOT_IMPL = pytest.mark.xfail(
    reason="TODO: `ArrowExpr.is_{duplicated,unique}`", raises=NotImplementedError
)

XFAIL_NOT_GROUP_BY = pytest.mark.xfail(
    reason="TODO: `ArrowExpr.is_{duplicated,unique}.over(*partition_by)`",
    raises=InvalidOperationError,
)


@pytest.fixture
def data() -> Data:
    return {
        "v1": [None, 2, 1, 4, 1],
        "v2": ["a", "c", "c", None, None],
        "p1": [2, 2, 2, 1, 1],
        "i": [0, 1, 2, 3, 4],
    }


@XFAIL_NOT_IMPL
def test_is_duplicated_unique(data: Data) -> None:
    expected = {
        "v1_is_unique": [True, True, False, True, False],
        "v2_is_unique": [True, False, False, False, False],
        "v1_is_duplicated": [False, False, True, False, True],
        "v2_is_duplicated": [False, True, True, True, True],
    }
    vals = nwp.col("v1", "v2")
    exprs = (
        vals.is_unique().name.suffix("_is_unique"),
        vals.is_duplicated().name.suffix("_is_duplicated"),
    )
    result = dataframe(data).select("i", *exprs).sort("i").drop("i")
    assert_equal_data(result, expected)  # pragma: no cover


# NOTE: Not supported on `main`
# Planning to adapt `is_{first,last}_distinct` idea here
@XFAIL_NOT_GROUP_BY
def test_is_duplicated_unique_partitioned(data: Data) -> None:
    expected = {
        "v1_is_unique": [True, True, True, True, True],
        "v2_is_unique": [True, False, False, False, False],
        "v1_is_duplicated": [False, False, False, False, False],
        "v2_is_duplicated": [False, True, True, True, True],
    }
    vals = ncs.by_index(0, 1)
    exprs = (
        vals.is_unique().name.suffix("_is_unique").over("p1"),
        vals.is_duplicated().name.suffix("_is_duplicated").over("p1"),
    )
    result = dataframe(data).select("i", *exprs).sort("i").drop("i")
    assert_equal_data(result, expected)  # pragma: no cover
