from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
from narwhals.stable.v1.dependencies import is_narwhals_dataframe

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_is_narwhals_dataframe(constructor_eager: ConstructorEager) -> None:
    df = constructor_eager({"col1": [1, 2], "col2": [3, 4]})

    assert is_narwhals_dataframe(nw.from_native(df))
    assert not is_narwhals_dataframe(df)
