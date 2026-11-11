from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw

pytest.importorskip("pyarrow")
import pyarrow as pa

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


@pytest.mark.filterwarnings("ignore:.*is_sparse is deprecated:DeprecationWarning")
def test_to_arrow(constructor_eager: ConstructorEager) -> None:
    data: dict[str, Any] = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]}
    df_raw = constructor_eager(data)
    result = nw.from_native(df_raw, eager_only=True).to_arrow()

    expected = pa.table(data)
    assert result == expected


def test_to_arrow_drops_index(constructor_eager: ConstructorEager) -> None:
    # A pandas-like index is not data, so it must not become a column.
    data = {"a": [1, 2, 3], "i": [5, 1, 9]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = nw.maybe_set_index(df, index=df["i"]).to_arrow()
    assert result == pa.table(data)
