from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from narwhals._plan.typing import IntoExpr
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "a": [[2, 2, 3, None, None], None, [], [None]],
        "b": [[1, 2, 2], [3, 4], [5, 5, 5, 6], [7]],
        "c": [1, 3, None, 2],
        "d": ["B", None, "A", "C"],
    }


a = nwp.col("a")
b = nwp.col("b")


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (2, [True, None, False, False]),
        (4, [False, None, False, False]),
        (nwp.col("c").last() + 1, [True, None, False, False]),
        (nwp.lit(None, nw.Int32), [True, None, False, True]),
    ],
)
def test_list_contains(data: Data, item: IntoExpr, expected: list[bool | None]) -> None:
    df = dataframe(data).with_columns(a.cast(nw.List(nw.Int32)))
    result = df.select(a.list.contains(item))
    assert_equal_data(result, {"a": expected})


R1: Final[list[Any]] = [None, "A", "B", "A", "A", "B"]
R2: Final = None
R3: Final[list[Any]] = []
R4: Final = [None]


@pytest.mark.xfail(
    reason=" TODO: `ArrowScalar.list.contains`", raises=NotImplementedError
)
@pytest.mark.parametrize(
    ("row", "item", "expected"),
    [
        (R1, "A", True),
        (R2, "A", None),
        (R3, "A", False),
        (R4, "A", False),
        (R1, None, True),
        (R2, None, None),
        (R3, None, False),
        (R4, None, True),
        (R1, "C", False),
        (R2, "C", None),
        (R3, "C", False),
        (R4, "C", False),
    ],
)
def test_list_contains_scalar(
    row: list[str | None] | None,
    item: IntoExpr,
    expected: bool | None,  # noqa: FBT001
) -> None:  # pragma: no cover
    data = {"a": [row]}
    df = dataframe(data).select(a.cast(nw.List(nw.String)))
    result = df.select(a.first().list.contains(item))
    assert_equal_data(result, {"a": expected})
