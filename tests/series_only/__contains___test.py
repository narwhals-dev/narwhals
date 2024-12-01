from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = [100, 200, None]


@pytest.mark.parametrize(("other", "expected"), [(100, True), (None, True), (3, False)])
def test_contains(
    constructor_eager: ConstructorEager,
    other: int | None,
    expected: bool,  # noqa: FBT001
) -> None:
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]

    assert (other in s) == expected
