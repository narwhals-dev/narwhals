from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import Constructor


def test_to_native(nw_frame_constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]}
    df_raw = nw_frame_constructor(data)
    df = nw.from_native(df_raw)

    assert isinstance(df.to_native(), df_raw.__class__)
