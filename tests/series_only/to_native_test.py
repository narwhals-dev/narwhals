from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw_unstable
import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = [4, 4, 4, 1, 6, 6, 4, 4, 1, 1]


def test_to_native(constructor_eager: ConstructorEager) -> None:
    orig_series = constructor_eager({"a": data})["a"]  # type: ignore[index]
    nw_series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    assert isinstance(nw_series.to_native(), orig_series.__class__)
    assert isinstance(nw_series.native, orig_series.__class__)


def test_raise_warning(constructor_eager: ConstructorEager) -> None:
    orig_series = constructor_eager({"a": data})["a"]  # type: ignore[index]
    nw_series = nw_unstable.from_native(constructor_eager({"a": data}), eager_only=True)[
        "a"
    ]

    with pytest.deprecated_call():
        assert isinstance(nw_series.to_native(), orig_series.__class__)
