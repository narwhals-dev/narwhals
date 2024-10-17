from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import assert_equal_data

data = [1, 2, 3]


def test_iter(constructor_eager: Any, request: pytest.FixtureRequest) -> None:
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]

    assert isinstance(s, Iterable)
    assert_equal_data({"a": [x for x in s]}, {"a": [1, 2, 3]})  # noqa: C416
