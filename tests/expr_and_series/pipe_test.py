from typing import Any

import narwhals as nw
from tests.utils import compare_dicts


@nw.narwhalify()
def func(s: Any) -> Any:
    return s.pipe(lambda x: x**2)


input_list = {"a": [2, 4, 6, 8]}
expected = [i**2 for i in input_list["a"]]


def test_pipe_expr(constructor: Any) -> None:
    df = nw.from_native(constructor(input_list))
    e = df.select(nw.col("a").pipe(func))
    compare_dicts(e, {"a": expected})


def test_pipe_series(
    constructor_series: Any,
) -> None:
    s = nw.from_native(constructor_series(input_list["a"]), series_only=True)
    assert s.pipe(func).to_list() == expected
