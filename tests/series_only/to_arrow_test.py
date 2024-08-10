from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw


def test_to_arrow(constructor_eager: Any) -> None:
    data = [1, 2, 3]
    result = nw.from_native(constructor_eager({"a": data}), eager_only=True)[
        "a"
    ].to_arrow()

    assert pa.types.is_int64(result.type)
    assert result == pa.array(data, type=pa.int64())


def test_to_arrow_with_nulls(constructor_eager: Any, request: Any) -> None:
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
    assert result == pa.array(data, type=pa.int64())
