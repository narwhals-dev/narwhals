from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from narwhals import _plan as nwp
from narwhals._plan import selectors as ncs
from narwhals.exceptions import ColumnNotFoundError, InvalidOperationError
from tests.plan.utils import dataframe, re_compile

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}


def test_unique_invalid_keep(data: Data) -> None:
    with pytest.raises(NotImplementedError, match=re_compile(r"found.+cabbage")):
        dataframe(data).unique(keep="cabbage")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("subset", "err"),
    [
        ("cabbage", ColumnNotFoundError),
        (ncs.string(), ColumnNotFoundError),
        (nwp.nth(5).meta.as_selector(), ColumnNotFoundError),
        (["a", "b", "A"], ColumnNotFoundError),
        (nwp.col("a").first(), InvalidOperationError),
        pytest.param(
            ncs.first().last(),
            InvalidOperationError,
            # TODO @dangotbanned: Fix this in another PR
            # Need to be stricter on the Selector check
            marks=pytest.mark.xfail(
                reason="narwhals/_plan/_expansion.py:160: 'Last' object has no attribute 'iter_expand_names'",
                raises=AttributeError,
            ),
        ),
    ],
)
def test_unique_invalid_subset(data: Data, subset: Any, err: type[Exception]) -> None:
    df = dataframe(data)
    with pytest.raises(err):
        df.unique(subset)
