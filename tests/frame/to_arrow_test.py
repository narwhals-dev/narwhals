from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pyarrow as pa
import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


@pytest.mark.filterwarnings("ignore:.*is_sparse is deprecated:DeprecationWarning")
def test_to_arrow(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
) -> None:
    if "pandas" in str(constructor_eager) and PANDAS_VERSION < (1, 0, 0):
        # pyarrow requires pandas>=1.0.0
        request.applymarker(pytest.mark.xfail)

    data: dict[str, Any] = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]}
    df_raw = constructor_eager(data)
    result = nw.from_native(df_raw, eager_only=True).to_arrow()

    expected = pa.table(data)
    assert result == expected
