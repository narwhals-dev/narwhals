from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw


def test_zip_with(request: Any, constructor_eager: Any) -> None:
    if "modin" in str(constructor_eager):
        msg = "modin has a known issue with casting #7364"
        request.applymarker(pytest.mark.xfail(reason=msg))
    series1 = nw.from_native(constructor_eager({"a": [1, 3, 2]}), eager_only=True)["a"]
    series2 = nw.from_native(constructor_eager({"a": [4, 4, 6]}), eager_only=True)["a"]
    mask = nw.from_native(constructor_eager({"a": [True, False, True]}), eager_only=True)[
        "a"
    ]

    result = series1.zip_with(mask, series2)
    expected = [1, 4, 2]
    assert result.to_list() == expected
