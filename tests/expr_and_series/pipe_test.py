from typing import Any

import narwhals as nw
from tests.utils import compare_dicts


@nw.narwhalify()
def func(s: Any) -> Any:
    return s.pipe(lambda x: x**2)


input_list = {"a": [2, 4, 6, 8]}
expected = [4, 16, 36, 64]


def test_pipe_expr(constructor: Any) -> None:
    df = nw.from_native(constructor(input_list))
    e = df.select(func(nw.col("a")))
    compare_dicts(e, {"a": expected})


def test_pipe_series(
    constructor_series: Any,
) -> None:
    s = nw.from_native(constructor_series(input_list["a"]), series_only=True)
    assert func(s).to_list() == expected
