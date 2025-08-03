from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


@pytest.mark.parametrize("n", [0, 1, 10])
def test_clear(constructor_eager: ConstructorEager, n: int) -> None:
    data = {
        "int": [1, 2, 3],
        "str": ["foo", "bar", "baz"],
        "float": [0.1, 0.2, 0.3],
        "bool": [True, False, True],
    }
    df = nw.from_native(constructor_eager(data), eager_only=True)
    df_clear = df.clear(n=n)

    assert len(df_clear) == n
    assert df.schema == df_clear.schema

    assert_equal_data(df_clear, {k: [None] * n for k in data})


def test_clear_negative(constructor_eager: ConstructorEager) -> None:
    n = -1
    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor_eager(data), eager_only=True)

    msg = f"`n` should be greater than or equal to 0, got {n}"
    with pytest.raises(ValueError, match=msg):
        df.clear(n=n)
