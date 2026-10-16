from __future__ import annotations

from datetime import time
from typing import TYPE_CHECKING

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import (
    NOT_INSTALLED,
    PANDAS_VERSION,
    POLARS_VERSION,
    PYARROW_VERSION,
    assert_equal_data,
    is_windows,
    maybe_collect,
)

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": ["12:34:56"]}


def requires_time_support(
    request: pytest.FixtureRequest, constructor: Constructor | ConstructorEager
) -> None:
    """Enforce Time dtype test expectations for dataframe backends.

    Skip or mark tests as expected failures depending on backend capabilities,
    version, and pyarrow availability when testing Time dtype support.
    """
    if constructor.__name__.startswith(("pandas", "modin")):
        if PANDAS_VERSION < PkgVersion("2.2"):
            pytest.skip(
                "pandas < 2.2.0 has no pyarrow dtype support (and therefore does not support the Time dtype)"
            )

        if PYARROW_VERSION == NOT_INSTALLED:
            pytest.skip("pandas requires pyarrow for the Time dtype")

    if "pyspark" in str(constructor) or "dask" in str(constructor):
        request.applymarker(
            pytest.mark.xfail(reason="backend does not support Time dtype")
        )


def test_to_time(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    requires_time_support(request, constructor)

    expected = "12:34:56"

    result = (
        nw.from_native(constructor(data))
        .lazy()
        .select(b=nw.col("a").str.to_time(format="%H:%M:%S"))
        .collect()
    )
    assert isinstance(result.collect_schema()["b"], nw.Time)
    assert str(result.item(row=0, column="b")) == expected


def test_to_time_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    requires_time_support(request, constructor_eager)

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
def test_to_time_infer_fmt(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    data: dict[str, list[str]],
    expected: str,
) -> None:
    requires_time_support(request, constructor)

    if (
        "polars" in str(constructor)
        and POLARS_VERSION < PkgVersion("1.30")
        and data["a"][0].count(":") < 2
    ):  # pragma: no cover
        request.applymarker(
            pytest.mark.xfail(reason="Polars<1.30 cannot auto-infer the HH:MM format")
        )

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
def test_to_time_series_infer_fmt(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    expected: str,
) -> None:
    requires_time_support(request, constructor_eager)

    if (
        "polars" in str(constructor_eager)
        and POLARS_VERSION < PkgVersion("1.30")
        and data["a"][0].count(":") < 2
    ):  # pragma: no cover
        request.applymarker(
            pytest.mark.xfail(reason="Polars<1.30 cannot auto-infer the HH:MM format")
        )

    result = nw.from_native(constructor_eager(data), eager_only=True)["a"].str.to_time()
    assert str(result.item(0)) == expected
    assert isinstance(result.dtype, nw.Time)


@pytest.mark.parametrize(
    ("data", "fmt", "expected"),
    [
        pytest.param({"a": ["12-34", None]}, "%H-%M", time(12, 34), id="dash"),
        pytest.param({"a": ["01:30 PM", None]}, "%I:%M %p", time(13, 30), id="ampm"),
    ],
)
def test_to_time_custom_format(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    data: dict[str, list[str | None]],
    fmt: str,
    expected: time,
) -> None:
    requires_time_support(request, constructor)
    if "%p" in fmt and is_windows() and "pyarrow_table" in str(constructor):
        reason = (
            "pyarrow cannot parse %p on Windows, see "
            "https://github.com/apache/arrow/issues/31971"
        )
        request.applymarker(pytest.mark.xfail(reason=reason))
    result = (
        nw.from_native(constructor(data)).lazy().select(b=nw.col("a").str.to_time(fmt))
    )
    assert isinstance(result.collect_schema()["b"], nw.Time)
    assert_equal_data(result, {"b": [expected, None]})


@pytest.mark.parametrize(
    ("data", "fmt"), [({"a": ["12:34"]}, "%H:%M:%S"), ({"a": ["25:00:00"]}, None)]
)
def test_to_time_invalid_raises(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    data: dict[str, list[str]],
    fmt: str | None,
) -> None:
    # These raise for any input, so the test would pass without parsing anything.
    if "pyspark" in str(constructor) or "dask" in str(constructor):
        pytest.skip("backend does not support Time dtype")
    requires_time_support(request, constructor)
    df = nw.from_native(constructor(data))
    expr = nw.col("a").str.to_time(fmt)
    # Broad `Exception`: error types differ per backend.
    with pytest.raises(Exception):  # noqa: B017, PT011
        maybe_collect(df.select(expr))
