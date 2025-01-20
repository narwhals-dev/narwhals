from __future__ import annotations

import pytest

from narwhals.stable.v1.dependencies import is_pandas_index


def test_is_pandas_index() -> None:
    pytest.importorskip("pandas")

    import pandas as pd

    data = [1, 2]
    s_pd = pd.Series(data)
    assert is_pandas_index(s_pd.index)
    assert not is_pandas_index(data)
