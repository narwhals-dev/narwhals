from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, POLARS_VERSION

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": ["12:34:56"]}


def is_pandaslike_without_pyarrow(constructor: Constructor | ConstructorEager) -> bool:
    """Returns True for pandas constructor that does not specify pyarrow and pyarrow is not importable.

    Testing environments that do have pandas but not pyarrow available should to xfail to .str.to_time.

    pandas does not natively support the Time datatype. As such, Narwhals
    attempts to automatically convert pandas series to a pyarrow-backed pandas
    series if pyarrow is available.
    """
    name = constructor.__name__
    return (
        name.startswith(("pandas", "modin"))
        and ("pyarrow" not in name)
        and (find_spec("pyarrow") is None)
    )


@pytest.mark.skipif(PANDAS_VERSION < (2, 2, 0), reason="pyarrow dtype not available")
def test_to_time(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if is_pandaslike_without_pyarrow(constructor) or (
        "pyspark" in str(constructor) or "dask" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)

    expected = "12:34:56"

    result = (
        nw.from_native(constructor(data))
        .lazy()
        .select(b=nw.col("a").str.to_time(format="%H:%M:%S"))
        .collect()
    )
    assert isinstance(result.collect_schema()["b"], nw.Time)
    assert str(result.item(row=0, column="b")) == expected


@pytest.mark.skipif(PANDAS_VERSION < (2, 2, 0), reason="pyarrow dtype not available")
def test_to_time_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if (
        is_pandaslike_without_pyarrow(constructor_eager)
        or "pyspark" in str(constructor_eager)
        or "dask" in str(constructor_eager)
    ):
        request.applymarker(pytest.mark.xfail)
    expected = "12:34:56.000000000" if "cudf" in str(constructor_eager) else "12:34:56"
    result = nw.from_native(constructor_eager(data), eager_only=True)["a"].str.to_time(
        format="%H:%M:%S"
    )

    assert isinstance(result.dtype, nw.Time)
    assert str(result.item(0)) == expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [({"a": ["12:34:56"]}, "12:34:56"), ({"a": ["12:34"]}, "12:34:00")],
)
@pytest.mark.skipif(PANDAS_VERSION < (2, 2, 0), reason="pyarrow dtype not available")
def test_to_time_infer_fmt(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    data: dict[str, list[str]],
    expected: str,
) -> None:
    if (
        (
            "polars" in str(constructor)
            and POLARS_VERSION < (1, 30)
            and data["a"][0].count(":") < 2
        )
        or is_pandaslike_without_pyarrow(constructor)
        or "pyspark" in str(constructor)
        or "dask" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)

    result = (
        nw.from_native(constructor(data))
        .lazy()
        .select(b=nw.col("a").str.to_time())
        .collect()
    )
    assert str(result.item(row=0, column="b")) == expected
    assert isinstance(result.collect_schema()["b"], nw.Time)


@pytest.mark.parametrize(
    ("data", "expected"),
    [({"a": ["12:34:56"]}, "12:34:56"), ({"a": ["12:34"]}, "12:34:00")],
)
@pytest.mark.skipif(PANDAS_VERSION < (2, 2, 0), reason="pyarrow dtype not available")
def test_to_time_series_infer_fmt(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    expected: str,
) -> None:
    if (
        (
            "polars" in str(constructor_eager)
            and POLARS_VERSION < (1, 30)
            and data["a"][0].count(":") < 2
        )
        or is_pandaslike_without_pyarrow(constructor_eager)
        or "pyspark" in str(constructor_eager)
    ):
        request.applymarker(pytest.mark.xfail)

    result = nw.from_native(constructor_eager(data), eager_only=True)["a"].str.to_time()
    assert str(result.item(0)) == expected
    assert isinstance(result.dtype, nw.Time)
