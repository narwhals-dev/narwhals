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

_EXPECTED_A = 0, 0, 2, -1, 1

data = {
    "a": _EXPECTED_A,
    "b": [1, 3, 2, None, None],
    "c": [None, None, 12.0, 5.5, 12.1],
    "d": [4, 0, 1, 3, 2],
    "group_1": ["C", "D", "D", "E", "E"],
    "group_2": [None, "G", "F", "G", None],
}


@pytest.mark.parametrize(
    ("cols", "by", "descending", "nulls_last", "expected"),
    [
        ("group_1", "d", False, False, {"group_1": ["D", "D", "E", "E", "C"]}),
        ("group_1", ["c", "b"], False, True, {"group_1": ["E", "D", "E", "C", "D"]}),
    ],
)
def test_sort_by(
    constructor_eager: ConstructorEager,
    cols: str | Sequence[str],
    by: str | Sequence[str],
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
        nw.col(cols).sort_by(by, descending=descending, nulls_last=nulls_last)
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, {"a": _EXPECTED_A, "b": [3, 2, 1, None, None]}),
        (True, False, {"a": _EXPECTED_A, "b": [None, None, 3, 2, 1]}),
        (False, True, {"a": _EXPECTED_A, "b": [1, 2, 3, None, None]}),
        (False, False, {"a": _EXPECTED_A, "b": [None, None, 1, 2, 3]}),
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
