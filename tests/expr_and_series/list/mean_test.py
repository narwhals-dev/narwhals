from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[3, 2, 2, 4, None], [-1]]}


def test_mean_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(
        backend in str(constructor) for backend in ("dask", "modin", "cudf", "sqlframe")
    ):
        request.applymarker(pytest.mark.xfail)
    result = (
        nw.from_native(constructor(data))
        .select(nw.col("a").cast(nw.List(nw.Int32())).list.mean())
        .lazy()
        .collect()["a"]
        .to_list()
    )
    assert result[0] == 2.75
    assert result[1] == -1


def test_mean_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(backend in str(constructor_eager) for backend in ("modin", "cudf")):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.mean().to_list()
    assert result[0] == 2.75
    assert result[1] == -1
