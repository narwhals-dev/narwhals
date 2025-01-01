from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


@pytest.mark.filterwarnings("ignore:.*Passing a BlockManager.*:DeprecationWarning")
@pytest.mark.skipif(
    PANDAS_VERSION < (2, 0, 0),
    reason="too old for pandas-pyarrow",
)
def test_convert_pandas(
    constructor_eager: ConstructorEager,
) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df_raw = constructor_eager(data)
    result = nw.from_native(df_raw).to_pandas()  # type: ignore[union-attr]

    if constructor_eager.__name__.startswith("pandas"):
        expected = constructor_eager(data)
    elif "modin[pyarrow]" in str(constructor_eager):
        expected = pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow")
    else:
        expected = pd.DataFrame(data)

    pd.testing.assert_frame_equal(result, expected)
