from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals._utils import Implementation
from narwhals.testing.constructors import (
    FrameConstructor,
    get_constructor,
    prepare_constructors,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    PropertyName: TypeAlias = str
    TrueNames: TypeAlias = set[str]
    FalseNames: TypeAlias = set[str]


def test_eager_returns_eager_frame() -> None:
    c = get_constructor("pandas")
    if not c.is_available:
        pytest.skip()

    df = nw.from_native(c({"x": [1, 2, 3]}))
    assert isinstance(df, nw.DataFrame)


def test_lazy_returns_lazy_frame() -> None:
    c = get_constructor("polars[lazy]")
    if not c.is_available:
        pytest.skip()

    lf = nw.from_native(c({"x": [1, 2, 3]}))
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
    ("is_non_nullable", {"pandas", "modin", "dask"}, {"polars[eager]"}),
]


@pytest.mark.parametrize(("prop", "true_names", "false_names"), _IS_PROPERTY_CASES)
def test_constructor_is_properties(
    prop: str, true_names: TrueNames, false_names: FalseNames
) -> None:
    for name in true_names:
        c = get_constructor(name)
        assert getattr(c, prop), f"{name}.{prop} should be True"
    for name in false_names:
        c = get_constructor(name)
        assert not getattr(c, prop), f"{name}.{prop} should be False"


def test_constructor_implementation() -> None:
    assert get_constructor("pandas").implementation is Implementation.PANDAS
    assert get_constructor("pandas[pyarrow]").implementation is Implementation.PANDAS
    assert get_constructor("polars[eager]").implementation is Implementation.POLARS
    assert (
        get_constructor("pyspark[connect]").implementation
        is Implementation.PYSPARK_CONNECT
    )


def test_constructor_dunder() -> None:
    c1 = get_constructor("pandas")
    c2 = get_constructor("pandas")
    assert c1.identifier == "pandas"
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert c1 != get_constructor("polars[eager]")
    assert c1 != "not a constructor"


def test_init_subclass_requires_implementation() -> None:
    with pytest.raises(TypeError, match="missing `implementation`"):

        class _BadConstructor(FrameConstructor, requirements=("polars",)):
            name = "polars[eager]"

            def __call__(self, obj: object, /, **kwds: object) -> None:  # type: ignore[override]
                ...  # pragma: no cover


def test_get_constructor() -> None:
    assert get_constructor("pandas[pyarrow]") == get_constructor("pandas[pyarrow]")


def test_get_constructor_invalid_name() -> None:
    with pytest.raises(ValueError, match="Unknown constructor"):
        get_constructor("not_a_backend")


def test_prepare_constructors_exclude_only() -> None:
    result = prepare_constructors(exclude=["pandas"])
    names = {c.name for c in result}
    assert "pandas" not in names
