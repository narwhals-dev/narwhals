import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

import narwhals as nw


def test_maybe_align_index() -> None:
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 0]))
    s = nw.from_native(pd.Series([1, 2, 3], index=[2, 1, 0]), series_only=True)
    result = nw.maybe_align_index(df, s)
    expected = pd.DataFrame({"a": [2, 1, 3]}, index=[2, 1, 0])
    assert_frame_equal(nw.to_native(result), expected)
    result = nw.maybe_align_index(s, df)
    expected = pd.Series([2, 1, 3], index=[1, 2, 0])
    assert_series_equal(nw.to_native(result), expected)
    result = nw.maybe_align_index(s, s.sort(descending=True))
    expected = pd.Series([3, 2, 1], index=[0, 1, 2])
    assert_series_equal(nw.to_native(result), expected)
    result = nw.maybe_align_index(df, df.sort("a", descending=True))
    expected = pd.DataFrame({"a": [3, 2, 1]}, index=[0, 2, 1])
    assert_frame_equal(nw.to_native(result), expected)
