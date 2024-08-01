from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw


@pytest.mark.parametrize(
    ("drop", "left"),
    [
        (["a"], ["b", "z"]),
        (["a", "b"], ["z"]),
    ],
)
def test_drop(constructor: Any, drop: list[str], left: list[str], request: Any) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    assert df.drop(drop).collect_schema().names() == left
    assert df.drop(*drop).collect_schema().names() == left
