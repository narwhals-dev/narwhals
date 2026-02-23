from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pytest

pytest.importorskip("polars")
import polars as pl

import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import selectors as ncs
from narwhals._utils import Implementation
from tests.plan.utils import dataframe, re_compile

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import TypeAlias

    from narwhals.typing import IntoBackend, LazyAllowed
    from tests.conftest import Data

    Outer = TypeVar("Outer")
    Inner = TypeVar("Inner")
    Constructor: TypeAlias = Callable[[Data], Outer]
    ConstructorLazy: TypeAlias = Callable[[Data], nwp.LazyFrame[Inner]]


@pytest.fixture
def data_small() -> dict[str, Any]:
    return {
        "a": ["A", "B", "A"],
        "b": [1, 2, 3],
        "c": [9, 2, 4],
        "d": [8, 7, 8],
        "e": [None, 9, 7],
        "f": [True, False, None],
        "g": [False, None, False],
        "h": [None, None, True],
        "i": [None, None, None],
        "j": [12.1, None, 4.0],
        "k": [42, 10, None],
        "l": [4, 5, 6],
        "m": [0, 1, 2],
        "n": ["dogs", "cats", None],
        "o": ["play", "swim", "walk"],
    }


@pytest.fixture(scope="session")
def lazyframe() -> ConstructorLazy[pl.LazyFrame]:
    pytest.importorskip("polars")
    import polars as pl

    def _(data: Data, /) -> nwp.LazyFrame[pl.LazyFrame]:
        return nwp.LazyFrame.from_native(pl.DataFrame(data).lazy())

    return _


def test_lazyframe_from_native(data_small: Data) -> None:
    pytest.importorskip("polars")
    import polars as pl

    pl_df = pl.DataFrame(data_small)
    pl_lf = pl_df.lazy()
    lf = nwp.LazyFrame.from_native(pl_lf)
    assert lf.implementation is nw.Implementation.POLARS
    pattern = re_compile(r"unsupported lazyframe.+polars.+dataframe")
    with pytest.raises(TypeError, match=pattern):
        # NOTE: Lacking a warning from the type checker is a known tradeoff
        nwp.LazyFrame.from_native(pl_df)
    pattern = re_compile(r"unsupported lazyframe.+polars.+series")
    with pytest.raises(TypeError, match=pattern):
        nwp.LazyFrame.from_native(pl_df.to_series())


def assert_equal_schema(
    result: nwp.LazyFrame[pl.LazyFrame], expected: pl.LazyFrame
) -> None:
    actual_schema = result.collect_schema()
    expected_schema = actual_schema.from_polars(expected.collect_schema())
    assert len(actual_schema) == len(expected_schema)
    assert actual_schema == expected_schema


def test_lazyframe_collect_schema(lazyframe: ConstructorLazy[pl.LazyFrame]) -> None:
    pytest.importorskip("polars")
    import polars as pl
    import polars.selectors as cs

    data: Data = {
        "a": [1, 2, 3],
        "b": [None, "four", "five"],
        "c": [dt.timedelta(6), dt.timedelta(7), None],
        "d": [8.9, 10.1112, 13.0],
    }
    lf = lazyframe(data)
    native = pl.LazyFrame(data)

    assert_equal_schema(lf, native)
    assert_equal_schema(lf.select("d", "a"), native.select("d", "a"))
    assert_equal_schema(
        lf.select(nwp.col("d").cast(nw.Float32), ncs.string().name.suffix("_suffix")),
        native.select(pl.col("d").cast(pl.Float32), cs.string().name.suffix("_suffix")),
    )
    assert_equal_schema(
        lf.drop(ncs.temporal()).with_columns(c="b"),
        native.drop(cs.temporal()).with_columns(c="b"),
    )

    assert_equal_schema(
        lf.filter(nwp.col("b").is_not_null())
        .rename({"b": "B"})
        .sort(ncs.by_name("B"))
        .with_columns(b=nwp.col("B").str.to_uppercase()),
        native.filter(pl.col("b").is_not_null())
        .rename({"b": "B"})
        .sort(cs.by_name("B"))
        .with_columns(b=pl.col("B").str.to_uppercase()),
    )


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (None, "same"),
        ("empty", "same"),
        ("polars", Implementation.POLARS),
        (Implementation.POLARS, Implementation.POLARS),
        (pl, Implementation.POLARS),
    ],
)
def test_dataframe_lazy(
    data_small: Data,
    backend: IntoBackend[LazyAllowed] | None | Literal["empty"],
    expected: Implementation | Literal["same"],
) -> None:
    pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    df = dataframe(data_small).select(nwp.nth(-1, -2, -3), "g")
    schema = nw.Schema(
        {"o": nw.String(), "n": nw.String(), "m": nw.Int64(), "g": nw.Boolean()}
    )
    assert df.schema == schema

    lazy = df.lazy() if backend == "empty" else df.lazy(backend)
    assert isinstance(lazy, nwp.LazyFrame)
    if expected == "same":
        assert lazy.implementation is df.implementation
    else:
        assert lazy.implementation is expected
    assert lazy.collect_schema() == schema


def test_dataframe_lazy_invalid(data_small: Data) -> None:
    pytest.importorskip("pyarrow")
    df = dataframe(data_small)
    pattern = re_compile("unsupported.+backend.+expected.+got.+pandas")
    with pytest.raises(TypeError, match=pattern):
        df.lazy("pandas")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=pattern):
        df.lazy(Implementation.PANDAS)  # type: ignore[arg-type]


@pytest.mark.xfail(reason="Not yet implemented", raises=NotImplementedError)
@pytest.mark.parametrize(
    "backend",
    [
        Implementation.DUCKDB,
        Implementation.DASK,
        Implementation.IBIS,
        Implementation.PYSPARK,
        Implementation.SQLFRAME,
        "duckdb",
        "dask",
        "ibis",
        "pyspark",
        "sqlframe",
    ],
)
def test_dataframe_lazy_todo(data_small: Data, backend: LazyAllowed) -> None:
    pytest.importorskip("pyarrow")
    impl = Implementation.from_backend(backend)
    pytest.importorskip(impl.name.lower())
    df = dataframe(data_small).select("e")
    assert df.lazy(backend).implementation is impl
