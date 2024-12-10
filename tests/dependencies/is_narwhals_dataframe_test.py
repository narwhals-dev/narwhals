from __future__ import annotations

import pandas as pd

import narwhals as nw
import narwhals.stable.v1 as nws
from narwhals.stable.v1.dependencies import is_narwhals_dataframe


def test_is_narwhals_dataframe() -> None:
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    assert is_narwhals_dataframe(nw.from_native(df))
    assert is_narwhals_dataframe(nws.from_native(df))
    assert not is_narwhals_dataframe(df)
