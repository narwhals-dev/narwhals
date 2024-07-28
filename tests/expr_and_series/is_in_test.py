from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

series = [1, 4, 2, 5]
data = {
    "a": series,
}


def test_expr_is_in(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").is_in([4, 5]))
    expected = {"a": [False, True, False, True]}

    compare_dicts(result, expected)


def test_ser_is_in(constructor_eager: Any) -> None:
    ser = nw.from_native(constructor_eager({"a": series}), eager_only=True)["a"]
    result = ser.is_in([4, 5]).to_list()
    assert not result[0]
    assert result[1]
    assert not result[2]
    assert result[3]
