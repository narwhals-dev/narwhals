import pandas as pd
import pytest

import narwhals as nw


def test_dupes() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    df1 = pd.DataFrame({"b": [1, 2, 3]})
    df = pd.concat([df, df1, df1], axis=1)
    with pytest.raises(ValueError, match="Expected unique"):
        nw.from_native(df)
