from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [1, 1, 1, 2, 2, 3], "b": [1, 2, 3, 4, 5, 6]}


def test_any_value_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))

    result = df.select(nw.col("a", "b").any_value())
    assert result.lazy().collect().shape == (1, 2)

    result = df.select(nw.col("a"), nw.col("b").any_value())
    assert result.lazy().collect().shape == (6, 2)

    result = df.select(nw.col("a").any_value(), nw.col("b"))
    assert result.lazy().collect().shape == (6, 2)


def test_any_value_group_by(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))

    result = df.group_by("a").agg(nw.col("b").any_value().alias("any"), nw.col("b").max())
    assert result.lazy().collect().shape == (3, 3)


def test_any_value_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
