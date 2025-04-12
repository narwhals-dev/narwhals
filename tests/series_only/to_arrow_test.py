from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_to_arrow(constructor_eager: ConstructorEager) -> None:
    data = [1, 2, 3]
    result = nw.from_native(constructor_eager({"a": data}), eager_only=True)[
        "a"
    ].to_arrow()

    assert pa.types.is_int64(result.type)
    assert pc.all(pc.equal(result, pa.array(data, type=pa.int64())))


def test_to_arrow_with_nulls(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "pandas_constructor" in str(constructor_eager) or "modin_constructor" in str(
        constructor_eager
    ):
        request.applymarker(pytest.mark.xfail)

    data = [1, 2, None]
    result = (
        nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
        .cast(nw.Int64)
        .to_arrow()
    )

    assert pa.types.is_int64(result.type)
    assert pc.all(pc.equal(result, pa.array(data, type=pa.int64())))
