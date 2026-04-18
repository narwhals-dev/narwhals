from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("pandas")
import pandas as pd

import narwhals as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


@pytest.mark.filterwarnings("ignore:.*Passing a BlockManager.*:DeprecationWarning")
@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="too old for pandas-pyarrow")
def test_convert_pandas(nw_eager_constructor: ConstructorEager) -> None:
    pytest.importorskip("pyarrow")
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = nw_eager_constructor(data)
    result = nw.from_native(df_raw, eager_only=True).to_pandas()

    if str(nw_eager_constructor).startswith("pandas"):
        expected = cast("pd.DataFrame", nw_eager_constructor(data))
    elif "modin_pyarrow" in str(nw_eager_constructor):
        expected = pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow")
    else:
        expected = pd.DataFrame(data)

    pd.testing.assert_frame_equal(result, expected)
