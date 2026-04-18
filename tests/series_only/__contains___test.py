from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = [100, 200, None]


@pytest.mark.parametrize(
    ("other", "expected"), [(100, True), (None, True), (1, False), (100.314, False)]
)
def test_contains(
    nw_eager_constructor: ConstructorEager,
    other: int | None,
    expected: bool,  # noqa: FBT001
) -> None:
    s = nw.from_native(nw_eager_constructor({"a": data}), eager_only=True)["a"]

    assert (other in s) == expected


@pytest.mark.parametrize("other", ["foo", [1, 2, 3]])
def test_contains_invalid_type(
    request: pytest.FixtureRequest, nw_eager_constructor: ConstructorEager, other: Any
) -> None:
    if "polars" not in str(nw_eager_constructor) and "pyarrow_table" not in str(
        nw_eager_constructor
    ):
        request.applymarker(pytest.mark.xfail)

    s = nw.from_native(nw_eager_constructor({"a": data}), eager_only=True)["a"]

    with pytest.raises(InvalidOperationError):
        _ = other in s
