from typing import Any

import narwhals as nw
from tests.utils import compare_dicts

input_list = {"a": [2, 4, 6, 8]}
expected = [4, 16, 36, 64]


def test_pipe_expr(constructor: Any) -> None:
    df = nw.from_native(constructor(input_list))
    e = df.select(nw.col("a").pipe(lambda x: x**2))
    compare_dicts(e, {"a": expected})


def test_pipe_series(
    constructor_eager: Any,
) -> None:
    s = nw.from_native(constructor_eager(input_list), eager_only=True)["a"]
    assert s.pipe(lambda x: x**2).to_list() == expected
