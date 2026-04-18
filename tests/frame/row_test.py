from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_row_column(request: Any, nw_eager_constructor: ConstructorEager) -> None:
    if "cudf" in str(nw_eager_constructor):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [11, 12, 13, 14, 15, 16]}
    result = nw.from_native(nw_eager_constructor(data), eager_only=True).row(2)
    assert result == (3.0, 13)
