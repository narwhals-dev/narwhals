from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager


def test_implode_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(backend in str(constructor_eager) for backend in ("cudf",)):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor_eager):
        pytest.importorskip("pyarrow")

    values = [1, 2, None, 3]
    series = nw.from_native(constructor_eager({"a": values}), eager_only=True)["a"]
    result = series.implode()
    expected = {"a": [values]}
    assert_equal_data({"a": result}, expected)


def test_implode_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(backend in str(constructor) for backend in ("cudf",)):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor):
        pytest.importorskip("pyarrow")

    values = [1, 2, None, 3]
    df = nw.from_native(constructor({"a": values, "idx": list(range(len(values)))}))
    result = df.sort("idx").select(nw.col("a").implode())

    expected = {"a": [values]}
    assert_equal_data(result, expected)


def test_implode_group_by(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(backend in str(constructor) for backend in ("cudf",)):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor):
        pytest.importorskip("pyarrow")

    data = {
        "idx": [0, 1, 2, 3, 4],
        "group": [1, 1, 2, 2, 3],
        "values": [2, 2, 3, None, None],
    }
    df = nw.from_native(constructor(data))
    result = (
        df.sort("idx").group_by("group").agg(nw.col("values").implode()).sort("group")
    )
    expected = {"group": [1, 2, 3], "values": [[2, 2], [3, None], [None]]}
    assert_equal_data(result, expected)
