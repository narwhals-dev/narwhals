from __future__ import annotations

import re

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"str": [["a", "b"], [None, "b"]]}
expected = {"str": ["a", None]}


def test_get_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(backend in str(constructor) for backend in ("dask", "modin", "cudf")):
        request.applymarker(pytest.mark.xfail)
    result = nw.from_native(constructor(data)).select(
        nw.col("str").cast(nw.List(nw.String())).list.get(0)
    )

    assert_equal_data(result, expected)


def test_get_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(backend in str(constructor_eager) for backend in ("modin", "cudf")):
        request.applymarker(pytest.mark.xfail)

    if (
        constructor_eager.__name__.startswith("pandas")
        and "pyarrow" not in constructor_eager.__name__
    ):
        df = nw.from_native(constructor_eager(data), eager_only=True)
        msg = re.escape("Series must be of PyArrow List type to support list namespace.")
        with pytest.raises(TypeError, match=msg):
            df["str"].list.get(0)
        return

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["str"].cast(nw.List(nw.String())).list.get(0)

    assert_equal_data({"str": result}, expected)
