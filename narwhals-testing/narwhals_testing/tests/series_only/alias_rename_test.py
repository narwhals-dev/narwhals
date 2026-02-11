from __future__ import annotations

from tests.utils import ConstructorEager, assert_equal_data

import narwhals as nw


def test_alias_rename(constructor_eager: ConstructorEager) -> None:
    data = [1, 2, 3]
    expected = {"bar": data}
    series = nw.from_native(constructor_eager({"foo": data}), eager_only=True)["foo"]
    result = series.alias("bar").to_frame()
    assert_equal_data(result, expected)
    result = series.rename("bar").to_frame()
    assert_equal_data(result, expected)
