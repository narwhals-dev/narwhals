from __future__ import annotations

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


def test_alias_rename(nw_eager_constructor: ConstructorEager) -> None:
    data = [1, 2, 3]
    expected = {"bar": data}
    series = nw.from_native(nw_eager_constructor({"foo": data}), eager_only=True)["foo"]
    result = series.alias("bar").to_frame()
    assert_equal_data(result, expected)
    result = series.rename("bar").to_frame()
    assert_equal_data(result, expected)
