import pandas as pd
import polars as pl

import narwhals as nw


def test_native_namespace() -> None:
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    assert nw.get_native_namespace(df) is pl
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))
    assert nw.get_native_namespace(df) is pd
