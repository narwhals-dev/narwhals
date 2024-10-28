from __future__ import annotations

import modin.pandas as mpd
import pandas as pd

from narwhals.dependencies import is_modin_index
from narwhals.dependencies import is_pandas_index
from narwhals.dependencies import is_pandas_like_index


def test_is_index() -> None:
    data = [1, 2]
    s_pd = pd.Series(data)
    s_md = mpd.Series(data)
    assert is_pandas_index(s_pd.index)
    assert is_modin_index(s_md.index)
    assert is_pandas_like_index(s_pd.index)
    assert is_pandas_like_index(s_md.index)
    assert not is_pandas_index(data)
    assert not is_modin_index(data)
