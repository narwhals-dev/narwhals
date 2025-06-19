from __future__ import annotations

import pytest

pytest.importorskip("pandas")
import pandas as pd


def test_stable_v1_dependenies_import() -> None:
    # https://github.com/bokeh/bokeh/pull/14530#issuecomment-2984474111
    # Note: for some reason, this test needs to be in its own file to reproduce
    # the error observed in Bokeh's CI. Please don't move this test to
    # tests/v1_test.py.
    import narwhals.stable.v1 as nw_v1

    df = nw_v1.from_native(pd.DataFrame({"a": [1, 2, 3]}))
    nw_v1.dependencies.is_pandas_dataframe(df)
