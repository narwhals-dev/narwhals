from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": ["12:34:56"]}


def test_to_time(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if (
        ("pandas" in str(constructor) and "pyarrow" not in str(constructor))
        or "pyspark" in str(constructor)
        or "dask" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    expected = "12:34:56"

    result = (
        nw.from_native(constructor(data))
        .lazy()
        .select(b=nw.col("a").str.to_time(format="%H:%M:%S"))
    )
    result_schema = result.collect_schema()
    assert isinstance(result_schema["b"], nw.Time)
    result_item = result.collect().item(row=0, column="b")
    assert str(result_item) == expected


def test_to_time_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if (
        ("pandas" in str(constructor_eager) and "pyarrow" not in str(constructor_eager))
        or "pyspark" in str(constructor_eager)
        or "dask" in str(constructor_eager)
    ):
        request.applymarker(pytest.mark.xfail)
    expected = "12:34:56.000000000" if "cudf" in str(constructor_eager) else "12:34:56"

    result = (
        nw.from_native(constructor_eager(data), eager_only=True)["a"].str.to_time(
            format="%H:%M:%S"
        )
    ).item(0)
    assert str(result) == expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [({"a": ["12:34:56"]}, "12:34:56"), ({"a": ["12:34"]}, "12:34:00")],
)
def test_to_time_infer_fmt(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    data: dict[str, list[str]],
    expected: str,
) -> None:
    if (
        ("polars" in str(constructor) and POLARS_VERSION < (1, 30) and data["a"][0].count(":") < 2)
        or ("pandas" in str(constructor) and "pyarrow" not in str(constructor))
        or "pyspark" in str(constructor)
        or "dask" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)

    result = (
        nw.from_native(constructor(data))
        .lazy()
        .select(b=nw.col("a").str.to_time())
        .collect()
        .item(row=0, column="b")
    )
    assert str(result) == expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [({"a": ["12:34:56"]}, "12:34:56"), ({"a": ["12:34"]}, "12:34:00")],
)
def test_to_time_series_infer_fmt(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    expected: str,
) -> None:
    if (
        ("polars" in str(constructor_eager) and POLARS_VERSION < (1, 30) and data["a"][0].count(":") < 2)
        or ("pandas" in str(constructor_eager) and "pyarrow" not in str(constructor_eager))
        or "pyspark" in str(constructor_eager)
    ):
        request.applymarker(pytest.mark.xfail)

    result = (
        nw.from_native(constructor_eager(data), eager_only=True)["a"].str.to_time()
    ).item(0)
    assert str(result) == expected
