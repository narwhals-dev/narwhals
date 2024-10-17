from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

input_list = {"a": [2, 4, 6, 8]}
expected = [4, 16, 36, 64]


def test_pipe_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(input_list))
    e = df.select(nw.col("a").pipe(lambda x: x**2))
    assert_equal_data(e, {"a": expected})


def test_pipe_series(
    constructor_eager: Any,
) -> None:
    s = nw.from_native(constructor_eager(input_list), eager_only=True)["a"]
    result = s.pipe(lambda x: x**2)
    assert_equal_data({"a": result}, {"a": expected})
