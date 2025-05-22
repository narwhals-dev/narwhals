from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_row_column(request: Any, constructor_eager: ConstructorEager) -> None:
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [11, 12, 13, 14, 15, 16]}
    result = nw.from_native(constructor_eager(data), eager_only=True).row(2)
    if "pyarrow_table" in str(constructor_eager):
        result = tuple(x.as_py() for x in result)
    assert result == (3.0, 13)
