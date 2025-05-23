from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


@pytest.mark.filterwarnings("ignore:.*Passing a BlockManager.*:DeprecationWarning")
@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="too old for pandas-pyarrow")
def test_convert_pandas(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = constructor_eager(data)
    result = nw.from_native(df_raw, eager_only=True).to_pandas()

    if constructor_eager.__name__.startswith("pandas"):
        expected = cast("pd.DataFrame", constructor_eager(data))
    elif "modin_pyarrow" in str(constructor_eager):
        expected = pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow")
    else:
        expected = pd.DataFrame(data)

    pd.testing.assert_frame_equal(result, expected)
