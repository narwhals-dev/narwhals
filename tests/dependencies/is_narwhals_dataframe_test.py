from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.stable.v1.dependencies import is_narwhals_dataframe

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_is_narwhals_dataframe(constructor_eager: ConstructorEager) -> None:
    df = constructor_eager({"col1": [1, 2], "col2": [3, 4]})

    assert is_narwhals_dataframe(nw.from_native(df))
    assert is_narwhals_dataframe(nw_v1.from_native(df))
    assert not is_narwhals_dataframe(df)
