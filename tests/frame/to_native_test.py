from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw_unstable
import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from tests.utils import Constructor


def test_to_native(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    assert isinstance(df.to_native(), df_raw.__class__)
    assert isinstance(df.native, df_raw.__class__)


def test_raise_warning(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]}
    df_raw = constructor(data)
    df = nw_unstable.from_native(df_raw)

    with pytest.deprecated_call():
        assert isinstance(df.to_native(), df_raw.__class__)
