import pandas as pd

import narwhals as nw


def test_narwhalify() -> None:
    @nw.narwhalify
    def func(df: nw.DataFrame) -> nw.DataFrame:
        return df.with_columns(nw.all() + 1)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))


def test_narwhalify_called() -> None:
    @nw.narwhalify()
    def func(df: nw.DataFrame) -> nw.DataFrame:
        return df.with_columns(nw.all() + 1)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))
