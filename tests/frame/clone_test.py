from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_clone(constructor_eager: ConstructorEager) -> None:
    expected = {"a": [1, 2], "b": [3, 4]}
    df = nw.from_native(constructor_eager(expected), eager_only=True)
    df_clone = df.clone()
    assert df is not df_clone
    assert df._compliant_frame is not df_clone._compliant_frame
    assert_equal_data(df_clone, expected)
