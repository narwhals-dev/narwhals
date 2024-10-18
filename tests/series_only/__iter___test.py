from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = [1, 2, 3]


def test_iter(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]

    assert isinstance(s, Iterable)
    compare_dicts({"a": [x for x in s]}, {"a": [1, 2, 3]})  # noqa: C416
