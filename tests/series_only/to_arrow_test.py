from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw

pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_to_arrow(nw_eager_constructor: ConstructorEager) -> None:
    data = [1, 2, 3]
    result = nw.from_native(nw_eager_constructor({"a": data}), eager_only=True)[
        "a"
    ].to_arrow()

    assert pa.types.is_int64(result.type)
    assert pc.all(pc.equal(result, pa.array(data, type=pa.int64())))


def test_to_arrow_with_nulls(
    nw_eager_constructor: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "pandas_constructor" in str(nw_eager_constructor) or "modin_constructor" in str(
        nw_eager_constructor
    ):
        request.applymarker(pytest.mark.xfail)

    data = [1, 2, None]
    result = (
        nw.from_native(nw_eager_constructor({"a": data}), eager_only=True)["a"]
        .cast(nw.Int64)
        .to_arrow()
    )

    assert pa.types.is_int64(result.type)
    assert pc.all(pc.equal(result, pa.array(data, type=pa.int64())))
