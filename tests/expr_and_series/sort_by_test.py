from __future__ import annotations

# ruff: noqa: FBT001
from typing import TYPE_CHECKING
from typing import Sequence

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import PythonLiteral
    from tests.utils import ConstructorEager

data = {"a": [0, 0, 2, -1], "b": [1, 3, 2, None]}


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, {"a": [0, 0, 2, -1], "b": [3, 2, 1, None]}),
        (True, False, {"a": [0, 0, 2, -1], "b": [None, 3, 2, 1]}),
        (False, True, {"a": [0, 0, 2, -1], "b": [1, 2, 3, None]}),
        (False, False, {"a": [0, 0, 2, -1], "b": [None, 1, 2, 3]}),
    ],
)
def test_sort_by_self(
    constructor_eager: ConstructorEager,
    descending: bool | Sequence[bool],
    nulls_last: bool,
    expected: dict[str, Sequence[PythonLiteral]],
    request: pytest.FixtureRequest,
) -> None:
    if any(
        x in str(constructor_eager) for x in ("pyarrow_table", "pandas", "modin", "cudf")
    ):
        request.applymarker(pytest.mark.xfail(reason="Not implemented"))
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        "a", nw.col("b").sort_by("b", descending=descending, nulls_last=nulls_last)
    )
    assert_equal_data(result, expected)
