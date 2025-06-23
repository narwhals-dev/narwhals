from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import Constructor, ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

data = {"abc": ["foo", "bars"], "xyz": [100, 200], "const": [42, 42]}


def test_with_row_index_eager(constructor_eager: ConstructorEager) -> None:
    result = nw.from_native(constructor_eager(data), eager_only=True).with_row_index()
    expected = {"index": [0, 1], **data}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("order_by", "expected_index"),
    [
        ("abc", [1, 0]),
        ("xyz", [0, 1]),
        (["const", "abc"], [1, 0]),
        (["const", "xyz"], [0, 1]),
    ],
)
def test_with_row_index_lazy(
    constructor: Constructor, order_by: str | Sequence[str], expected_index: list[int]
) -> None:
    result = (
        nw.from_native(constructor(data)).with_row_index(order_by=order_by).sort("xyz")
    )
    expected = {"index": expected_index, **data}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("namespace", [nw, nw_v1])
def test_with_row_index_lazy_exception(
    constructor: Constructor, namespace: ModuleType
) -> None:
    frame = namespace.from_native(constructor(data))

    context = (
        pytest.raises(Exception)  # noqa: PT011
        if any(x in str(constructor) for x in ("duckdb", "pyspark"))
        or (namespace is nw and isinstance(frame, namespace.LazyFrame))
        else does_not_raise()
    )

    with context:
        frame.with_row_index()


# TODO(FBruzzesi): Validate Dask and Ibis results with nw_v1 and order_by=None.
