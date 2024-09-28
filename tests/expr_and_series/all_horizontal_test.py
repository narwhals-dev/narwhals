from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import compare_dicts


@pytest.mark.parametrize("expr1", ["a", nw.col("a")])
@pytest.mark.parametrize("expr2", ["b", nw.col("b")])
def test_allh(constructor: Constructor, expr1: Any, expr2: Any) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor(data))
    result = df.select(all=nw.all_horizontal(expr1, expr2))

    expected = {"all": [False, False, True]}
    compare_dicts(result, expected)


def test_allh_series(constructor_eager: ConstructorEager) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(all=nw.all_horizontal(df["a"], df["b"]))

    expected = {"all": [False, False, True]}
    compare_dicts(result, expected)
