from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals._utils import Implementation
from narwhals.testing.constructors import (
    available_backends,
    get_backend_constructor,
    prepare_backends,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    PropertyName: TypeAlias = str
    TrueNames: TypeAlias = set[str]
    FalseNames: TypeAlias = set[str]


def test_eager_returns_eager_frame() -> None:
    c = get_backend_constructor("pandas")
    if not c.is_available:
        pytest.skip()

    df = c({"x": [1, 2, 3]}, nw)
    assert isinstance(df, nw.DataFrame)


def test_lazy_returns_lazy_frame() -> None:
    c = get_backend_constructor("polars[lazy]")
    if not c.is_available:
        pytest.skip()

    lf = c({"x": [1, 2, 3]}, nw)
    assert isinstance(lf, nw.LazyFrame)


_IS_PROPERTY_CASES: list[tuple[PropertyName, TrueNames, FalseNames]] = [
    ("is_pandas", {"pandas", "pandas[nullable]", "pandas[pyarrow]"}, {"polars[eager]"}),
    ("is_modin", {"modin", "modin[pyarrow]"}, {"pandas"}),
    ("is_cudf", {"cudf"}, {"pandas"}),
    ("is_pandas_like", {"pandas", "modin", "cudf"}, {"polars[eager]"}),
    ("is_polars", {"polars[eager]", "polars[lazy]"}, {"pandas"}),
    ("is_pyarrow", {"pyarrow"}, {"pandas"}),
    ("is_dask", {"dask"}, {"pandas"}),
    ("is_duckdb", {"duckdb"}, {"pandas"}),
    ("is_pyspark", {"pyspark", "pyspark[connect]"}, {"pandas"}),
    ("is_sqlframe", {"sqlframe"}, {"pandas"}),
    ("is_ibis", {"ibis"}, {"pandas"}),
    ("is_spark_like", {"pyspark", "sqlframe", "pyspark[connect]"}, {"pandas"}),
    ("is_lazy", {"polars[lazy]", "dask", "duckdb"}, {"pandas"}),
    ("needs_pyarrow", {"pyarrow", "duckdb", "ibis"}, {"pandas"}),
    ("is_nullable", {"polars[eager]"}, {"pandas", "modin", "dask"}),
]


@pytest.mark.parametrize(("prop", "true_names", "false_names"), _IS_PROPERTY_CASES)
def test_constructor_is_properties(
    prop: str, true_names: TrueNames, false_names: FalseNames
) -> None:
    for name in true_names:
        c = get_backend_constructor(name)
        assert getattr(c, prop), f"{name}.{prop} should be True"
    for name in false_names:
        c = get_backend_constructor(name)
        assert not getattr(c, prop), f"{name}.{prop} should be False"


def test_constructor_implementation() -> None:
    assert get_backend_constructor("pandas").implementation is Implementation.PANDAS
    assert (
        get_backend_constructor("pandas[pyarrow]").implementation is Implementation.PANDAS
    )
    assert (
        get_backend_constructor("polars[eager]").implementation is Implementation.POLARS
    )
    assert (
        get_backend_constructor("pyspark[connect]").implementation
        is Implementation.PYSPARK_CONNECT
    )


def test_constructor_dunder() -> None:
    c1 = get_backend_constructor("pandas")
    c2 = get_backend_constructor("pandas")
    assert c1.identifier == "pandas"
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert c1 != get_backend_constructor("polars[eager]")
    assert c1 != "not a constructor"


def test_get_backend_constructor_invalid_name() -> None:
    with pytest.raises(ValueError, match="Unknown constructor"):
        get_backend_constructor("not_a_backend")


@pytest.mark.parametrize(
    ("include", "exclude", "expected"),
    [
        (None, None, available_backends()),
        (None, ["pandas"], available_backends() - {"pandas"}),
        (["pandas", "polars[eager]"], None, {"pandas", "polars[eager]"}),
        (["pandas", "polars[eager]"], ["pandas"], {"polars[eager]"}),
        ([], None, frozenset()),
    ],
)
def test_prepare_backends(
    include: list[str] | None, exclude: list[str] | None, expected: frozenset[str]
) -> None:
    for name in (*(include or ()), *(exclude or ())):
        if not get_backend_constructor(name).is_available:
            pytest.skip(f"{name} not installed")
    result = prepare_backends(include=include, exclude=exclude)
    assert {c.name for c in result} == expected


@pytest.mark.parametrize("kwarg", ["include", "exclude"])
def test_prepare_backends_unknown_name_raises(kwarg: str) -> None:
    with pytest.raises(ValueError, match="not known constructors"):
        prepare_backends(**{kwarg: ["not_a_backend"]})
