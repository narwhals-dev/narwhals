from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw

pytest.importorskip("pyarrow")
import pyarrow as pa

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


@pytest.mark.filterwarnings("ignore:.*is_sparse is deprecated:DeprecationWarning")
def test_to_arrow(nw_eager_constructor: ConstructorEager) -> None:
    data: dict[str, Any] = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]}
    df_raw = nw_eager_constructor(data)
    result = nw.from_native(df_raw, eager_only=True).to_arrow()

    expected = pa.table(data)
    assert result == expected
