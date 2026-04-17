from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals._utils import Implementation
from narwhals.testing.constructors import (
    ConstructorName,
    get_constructor,
    prepare_constructors,
)
from narwhals.testing.constructors._classes import (
    ConstructorBase,
    DaskConstructor,
    PandasConstructor,
    PolarsEagerConstructor as OriginalPolarsEagerConstructor,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    PropertyName: TypeAlias = str
    ReturnTrue: TypeAlias = set[ConstructorName]
    ReturnFalse: TypeAlias = set[ConstructorName]


def test_dask_npartitions_distinct() -> None:
    dp1, dp2 = DaskConstructor(npartitions=1), DaskConstructor(npartitions=2)
    assert dp1 != dp2
    assert hash(dp1) != hash(dp2)


def test_dask_repr() -> None:
    assert repr(DaskConstructor(npartitions=3)) == "DaskConstructor(npartitions=3)"


def test_eager_returns_eager_frame() -> None:
    cn = ConstructorName.PANDAS
    if not cn.is_available:
        pytest.skip()

    df = nw.from_native(cn.constructor({"x": [1, 2, 3]}))
    assert isinstance(df, nw.DataFrame)


def test_lazy_returns_lazy_frame() -> None:
    cn = ConstructorName.POLARS_LAZY
    if not cn.is_available:
        pytest.skip()

    lf = nw.from_native(cn.constructor({"x": [1, 2, 3]}))
    assert isinstance(lf, nw.LazyFrame)


CN = ConstructorName


_IS_PROPERTY_CASES: list[tuple[PropertyName, ReturnTrue, ReturnFalse]] = [
    ("is_pandas", {CN.PANDAS, CN.PANDAS_NULLABLE, CN.PANDAS_PYARROW}, {CN.POLARS_EAGER}),
    ("is_modin", {CN.MODIN, CN.MODIN_PYARROW}, {CN.PANDAS}),
    ("is_cudf", {CN.CUDF}, {CN.PANDAS}),
    ("is_pandas_like", {CN.PANDAS, CN.MODIN, CN.CUDF}, {CN.POLARS_EAGER}),
    ("is_polars", {CN.POLARS_EAGER, CN.POLARS_LAZY}, {CN.PANDAS}),
    ("is_pyarrow", {CN.PYARROW}, {CN.PANDAS}),
    ("is_dask", {CN.DASK}, {CN.PANDAS}),
    ("is_duckdb", {CN.DUCKDB}, {CN.PANDAS}),
    ("is_pyspark", {CN.PYSPARK, CN.PYSPARK_CONNECT}, {CN.PANDAS}),
    ("is_sqlframe", {CN.SQLFRAME}, {CN.PANDAS}),
    ("is_ibis", {CN.IBIS}, {CN.PANDAS}),
    ("is_spark_like", {CN.PYSPARK, CN.SQLFRAME, CN.PYSPARK_CONNECT}, {CN.PANDAS}),
    ("is_lazy", {CN.POLARS_LAZY, CN.DASK, CN.DUCKDB}, {CN.PANDAS}),
    ("needs_pyarrow", {CN.PYARROW, CN.DUCKDB, CN.IBIS}, {CN.PANDAS}),
    ("is_non_nullable", {CN.PANDAS, CN.MODIN, CN.DASK}, {CN.POLARS_EAGER}),
]


@pytest.mark.parametrize(("prop", "true_names", "false_names"), _IS_PROPERTY_CASES)
def test_constructor_name_is_properties(
    prop: str, true_names: set[ConstructorName], false_names: set[ConstructorName]
) -> None:
    for name in true_names:
        assert getattr(name, prop), f"{name}.{prop} should be True"
    for name in false_names:
        assert not getattr(name, prop), f"{name}.{prop} should be False"


def test_constructor_name_implementation() -> None:
    assert CN.PANDAS.implementation is Implementation.PANDAS
    assert CN.PANDAS_PYARROW.implementation is Implementation.PANDAS
    assert CN.POLARS_EAGER.implementation is Implementation.POLARS
    assert CN.PYSPARK_CONNECT.implementation is Implementation.PYSPARK_CONNECT


def test_constructor_dunder() -> None:
    c1 = ConstructorName.PANDAS.constructor
    c2 = PandasConstructor()
    assert c1.identifier == "pandas"
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert c1 != OriginalPolarsEagerConstructor()
    assert c1 != "not a constructor"


def test_init_subclass_no_legacy_name() -> None:
    class _Dummy(
        ConstructorBase, implementation=Implementation.POLARS, requirements=("polars",)
    ):
        name = ConstructorName.POLARS_EAGER

        def __call__(self, obj: object, /, **kwds: object) -> None:  # type: ignore[override]
            ...  # pragma: no cover

    # re-registered POLARS_EAGER (overwriting the real one), but without a legacy_name for it.
    registered = ConstructorBase._registry[ConstructorName.POLARS_EAGER]
    assert registered == _Dummy()
    assert registered.legacy_name == ""
    assert registered.requirements == ("polars",)
    assert registered.implementation is Implementation.POLARS

    # Restore the original
    legacy_name = "polars_eager_constructor"

    class PolarsEagerConstructor(
        OriginalPolarsEagerConstructor,
        implementation=Implementation.POLARS,
        requirements=("polars",),
        legacy_name=legacy_name,
    ):
        name = ConstructorName.POLARS_EAGER

    original = PolarsEagerConstructor()

    restored = ConstructorBase._registry[ConstructorName.POLARS_EAGER]
    assert restored == original
    assert restored.legacy_name == legacy_name


def test_init_subclass_requires_implementation() -> None:
    with pytest.raises(TypeError, match="missing `implementation`"):

        class _BadConstructor(ConstructorBase, requirements=("polars",)):
            name = ConstructorName.POLARS_EAGER

            def __call__(self, obj: object, /, **kwds: object) -> None:  # type: ignore[override]
                ...  # pragma: no cover


def test_get_constructor() -> None:
    expected = ConstructorName.PANDAS_PYARROW.constructor
    assert get_constructor("pandas[pyarrow]") == expected


def test_get_constructor_invalid_name() -> None:
    with pytest.raises(ValueError, match="Unknown constructor"):
        get_constructor("not_a_backend")


def test_prepare_constructors_exclude_only() -> None:
    result = prepare_constructors(exclude=[ConstructorName.PANDAS])
    names = {c.name for c in result}
    assert ConstructorName.PANDAS not in names
