from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [float("nan"), float("inf"), 2.0]}
expected = {"a": [False, False, True]}


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_is_finite_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").is_finite())

    assert_equal_data(result, expected)


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_is_finite_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {"a": df["a"].is_finite()}

    assert_equal_data(result, expected)
