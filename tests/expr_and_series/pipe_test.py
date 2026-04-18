from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

input_list = {"a": [2, 4, 6, 8]}
expected = [4, 16, 36, 64]


def test_pipe_expr(nw_frame_constructor: Constructor) -> None:
    df = nw.from_native(nw_frame_constructor(input_list))
    e = df.select(nw.col("a").pipe(lambda x: x**2))
    assert_equal_data(e, {"a": expected})


def test_pipe_series(nw_eager_constructor: ConstructorEager) -> None:
    s = nw.from_native(nw_eager_constructor(input_list), eager_only=True)["a"]
    result = s.pipe(lambda x: x**2)
    assert_equal_data({"a": result}, {"a": expected})
