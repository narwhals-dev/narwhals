from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
from narwhals.stable.v1.dependencies import is_narwhals_series

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_is_narwhals_series(constructor_eager: ConstructorEager) -> None:
    df = constructor_eager({"col1": [1, 2], "col2": [3, 4]})

    assert is_narwhals_series(nw.from_native(df, eager_only=True)["col1"])
    assert not is_narwhals_series(nw.from_native(df, eager_only=True)["col1"].to_native())
