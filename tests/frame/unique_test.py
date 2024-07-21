from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}


@pytest.mark.parametrize("subset", ["b", ["b"]])
@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("first", {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}),
        ("last", {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}),
        ("any", {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}),
        ("none", {"a": [2], "b": [6], "z": [9]}),
    ],
)
def test_unique(
    constructor_with_lazy: Any,
    subset: str | list[str] | None,
    keep: str,
    expected: dict[str, list[float]],
) -> None:
    df_raw = constructor_with_lazy(data)
    df = nw.from_native(df_raw)

    result = df.unique(subset, keep=keep, maintain_order=True)  # type: ignore[arg-type]
    compare_dicts(result, expected)


def test_unique_none(constructor_with_lazy: Any) -> None:
    df_raw = constructor_with_lazy(data)
    df = nw.from_native(df_raw)

    result = df.unique(maintain_order=True)
    compare_dicts(result, data)
