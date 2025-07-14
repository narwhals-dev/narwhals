from __future__ import annotations

import re
import string
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

import hypothesis.strategies as st
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

import narwhals as nw
from narwhals._utils import (
    Implementation,
    Version,
    _DeferredIterable,
    check_columns_exist,
    deprecate_native_namespace,
    parse_version,
    requires,
)
from tests.utils import get_module_version_as_tuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from types import ModuleType

    from typing_extensions import Self

    from narwhals._utils import _SupportsVersion
    from narwhals.series import Series
    from narwhals.typing import IntoSeries


@dataclass
class DummyModule:
    __version__: str


def test_maybe_align_index_pandas() -> None:
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 0]))
    s = nw.from_native(pd.Series([1, 2, 3], index=[2, 1, 0]), series_only=True)
    result = nw.maybe_align_index(df, s)
    expected = pd.DataFrame({"a": [2, 1, 3]}, index=[2, 1, 0])
    assert_frame_equal(nw.to_native(result), expected)
    result = nw.maybe_align_index(df, df.sort("a", descending=True))
    expected = pd.DataFrame({"a": [3, 2, 1]}, index=[0, 2, 1])
    assert_frame_equal(nw.to_native(result), expected)
    result_s = nw.maybe_align_index(s, df)
    expected_s = pd.Series([2, 1, 3], index=[1, 2, 0])
    assert_series_equal(nw.to_native(result_s), expected_s)
    result_s = nw.maybe_align_index(s, s.sort(descending=True))
    expected_s = pd.Series([3, 2, 1], index=[0, 1, 2])
    assert_series_equal(nw.to_native(result_s), expected_s)


def test_with_columns_sort() -> None:
    # Check that, unlike in pandas, we don't change the index
    # when sorting
    df = nw.from_native(pd.DataFrame({"a": [2, 1, 3]}))
    result = df.with_columns(a_sorted=df["a"].sort()).pipe(nw.to_native)
    expected = pd.DataFrame({"a": [2, 1, 3], "a_sorted": [1, 2, 3]})
    assert_frame_equal(result, expected)


def test_non_unique_index() -> None:
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 0]))
    s = nw.from_native(pd.Series([1, 2, 3], index=[2, 2, 0]), series_only=True)
    with pytest.raises(ValueError, match="unique"):
        nw.maybe_align_index(df, s)


def test_maybe_align_index_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    s = nw.from_native(pl.Series([1, 2, 3]), series_only=True)
    result = nw.maybe_align_index(df, s)
    assert result is df
    with pytest.raises(ValueError, match="length"):
        nw.maybe_align_index(df, s[1:])


@pytest.mark.parametrize("column_names", ["b", ["a", "b"]])
def test_maybe_set_index_pandas_column_names(
    column_names: str | list[str] | None,
) -> None:
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = nw.maybe_set_index(df, column_names)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).set_index(column_names)
    assert_frame_equal(nw.to_native(result), expected)


@pytest.mark.parametrize("column_names", ["b", ["a", "b"]])
def test_maybe_set_index_polars_column_names(
    column_names: str | list[str] | None,
) -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = nw.maybe_set_index(df, column_names)
    assert result is df


@pytest.mark.parametrize(
    "native_df_or_series",
    [pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), pd.Series([0, 1, 2])],
)
@pytest.mark.parametrize(
    ("narwhals_index", "pandas_index"),
    [
        (nw.from_native(pd.Series([1, 2, 0]), series_only=True), pd.Series([1, 2, 0])),
        (
            [
                nw.from_native(pd.Series([0, 1, 2]), series_only=True),
                nw.from_native(pd.Series([1, 2, 0]), series_only=True),
            ],
            [pd.Series([0, 1, 2]), pd.Series([1, 2, 0])],
        ),
    ],
)
def test_maybe_set_index_pandas_direct_index(
    narwhals_index: Series[IntoSeries] | list[Series[IntoSeries]],
    pandas_index: pd.Series[Any] | list[pd.Series[Any]],
    native_df_or_series: pd.DataFrame | pd.Series[Any],
) -> None:
    df = nw.from_native(native_df_or_series, allow_series=True)
    result = nw.maybe_set_index(df, index=narwhals_index)
    if isinstance(native_df_or_series, pd.Series):
        native_df_or_series.index = pandas_index  # type: ignore[assignment]
        assert_series_equal(nw.to_native(result), native_df_or_series)
    else:
        expected = native_df_or_series.set_index(pandas_index)  # type: ignore[arg-type]
        assert_frame_equal(nw.to_native(result), expected)


@pytest.mark.parametrize(
    "index",
    [
        nw.from_native(pd.Series([1, 2, 0]), series_only=True),
        [
            nw.from_native(pd.Series([0, 1, 2]), series_only=True),
            nw.from_native(pd.Series([1, 2, 0]), series_only=True),
        ],
    ],
)
def test_maybe_set_index_polars_direct_index(
    index: Series[IntoSeries] | list[Series[IntoSeries]] | None,
) -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = nw.maybe_set_index(df, index=index)
    assert result is df


def test_maybe_set_index_pandas_series_column_names() -> None:
    df = nw.from_native(pd.Series([0, 1, 2]), allow_series=True)
    with pytest.raises(
        ValueError, match="Cannot set index using column names on a Series"
    ):
        nw.maybe_set_index(df, column_names=["a"])


def test_maybe_set_index_pandas_either_index_or_column_names() -> None:
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    column_names = ["a", "b"]
    index = nw.from_native(pd.Series([0, 1, 2]), series_only=True)
    with pytest.raises(
        ValueError, match="Only one of `column_names` or `index` should be provided"
    ):
        nw.maybe_set_index(df, column_names=column_names, index=index)
    with pytest.raises(
        ValueError, match="Either `column_names` or `index` should be provided"
    ):
        nw.maybe_set_index(df)


def test_maybe_get_index_pandas() -> None:
    pandas_df = pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 0])
    result = cast("pd.Index[Any]", nw.maybe_get_index(nw.from_native(pandas_df)))
    assert_index_equal(result, pandas_df.index)
    pandas_series = pd.Series([1, 2, 3], index=[1, 2, 0])
    result_s = cast(
        "pd.Index[Any]",
        nw.maybe_get_index(nw.from_native(pandas_series, series_only=True)),
    )
    assert_index_equal(result_s, pandas_series.index)


def test_maybe_get_index_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    result = nw.maybe_get_index(df)
    assert result is None
    series = nw.from_native(pl.Series([1, 2, 3]), series_only=True)
    result = nw.maybe_get_index(series)
    assert result is None


def test_maybe_reset_index_pandas() -> None:
    pandas_df = nw.from_native(
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[7, 8, 9])
    )
    result = nw.maybe_reset_index(pandas_df)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[0, 1, 2])
    assert_frame_equal(nw.to_native(result), expected)
    pandas_df = nw.from_native(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = nw.maybe_reset_index(pandas_df)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert_frame_equal(nw.to_native(result), expected)
    assert result.to_native() is pandas_df.to_native()
    pandas_series = nw.from_native(
        pd.Series([1, 2, 3], index=[7, 8, 9]), series_only=True
    )
    result_s = nw.maybe_reset_index(pandas_series)
    expected_s = pd.Series([1, 2, 3], index=[0, 1, 2])
    assert_series_equal(nw.to_native(result_s), expected_s)
    pandas_series = nw.from_native(pd.Series([1, 2, 3]), series_only=True)
    result_s = nw.maybe_reset_index(pandas_series)
    expected_s = pd.Series([1, 2, 3])
    assert_series_equal(nw.to_native(result_s), expected_s)
    assert result_s.to_native() is pandas_series.to_native()


def test_maybe_reset_index_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = nw.maybe_reset_index(df)
    assert result is df
    series = nw.from_native(pl.Series([1, 2, 3]), series_only=True)
    result_s = nw.maybe_reset_index(series)
    assert result_s is series


def test_maybe_convert_dtypes_pandas() -> None:
    import numpy as np

    df = nw.from_native(
        pd.DataFrame({"a": [1, np.nan]}, dtype=np.dtype("float64")), eager_only=True
    )
    result = nw.to_native(nw.maybe_convert_dtypes(df))
    expected = pd.DataFrame({"a": [1, pd.NA]}, dtype="Int64")
    pd.testing.assert_frame_equal(result, expected)
    result_s = nw.to_native(nw.maybe_convert_dtypes(df["a"]))
    expected_s = pd.Series([1, pd.NA], name="a", dtype="Int64")
    pd.testing.assert_series_equal(result_s, expected_s)


def test_maybe_convert_dtypes_polars() -> None:
    import numpy as np

    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame({"a": [1.1, np.nan]}))
    result = nw.maybe_convert_dtypes(df)
    assert result is df


def test_get_trivial_version_with_uninstalled_module() -> None:
    result = get_module_version_as_tuple("non_existent_module")
    assert result == (0, 0, 0)


@given(n_bytes=st.integers(1, 100))
@pytest.mark.slow
def test_generate_temporary_column_name(n_bytes: int) -> None:
    columns = ["abc", "XYZ"]

    temp_col_name = nw.generate_temporary_column_name(n_bytes=n_bytes, columns=columns)
    assert temp_col_name not in columns


def test_generate_temporary_column_name_raise() -> None:
    from itertools import product

    columns = [
        "".join(t)
        for t in product(
            string.ascii_lowercase + string.digits, string.ascii_lowercase + string.digits
        )
    ]

    with pytest.raises(
        AssertionError,
        match="Internal Error: Narwhals was not able to generate a column name with ",
    ):
        nw.generate_temporary_column_name(n_bytes=1, columns=columns)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2020.1.2", (2020, 1, 2)),
        ("2020.1.2-dev123", (2020, 1, 2)),
        ("3.0.0.dev0+618.gb552dc95c9", (3, 0, 0)),
        (DummyModule("2020.1.2-dev123"), (2020, 1, 2)),
    ],
)
def test_parse_version(
    version: str | _SupportsVersion, expected: tuple[int, ...]
) -> None:
    assert parse_version(version) == expected


def test_check_columns_exists() -> None:
    columns = ["a", "b", "c"]
    subset = ["d", "f"]
    error = check_columns_exist(subset, available=columns)
    assert error is not None
    assert str(error) == (
        "The following columns were not found: ['d', 'f']\n\nHint: Did you mean one of these columns: ['a', 'b', 'c']?"
    )

    # Check that the error is not returned
    subset = ["a", "b"]
    error = check_columns_exist(subset, available=columns)
    assert error is None


def test_not_implemented() -> None:
    pytest.importorskip("polars")

    from narwhals._arrow.expr import ArrowExpr
    from narwhals._polars.expr import PolarsExpr, PolarsExprStringNamespace
    from narwhals._utils import not_implemented

    data: dict[str, Any] = {"foo": [1, 2], "bar": [6.0, 7.0]}
    df = pa.table(data)
    nw_df = nw.from_native(df)
    ewm_mean = nw.col("foo").ewm_mean(com=1, ignore_nulls=False)
    pattern = re.compile(
        r".+ewm_mean.+ not implemented.+arrow", flags=re.DOTALL | re.IGNORECASE
    )
    with pytest.raises(NotImplementedError, match=pattern):
        nw_df.with_columns(ewm_mean)

    assert isinstance(ArrowExpr.ewm_mean, not_implemented)

    if TYPE_CHECKING:
        from narwhals._utils import _SupportsGet

    class DummyCompliant(Protocol):
        _implementation: nw.Implementation

        def alias(self, name: str) -> str: ...
        def unique(self) -> Self: ...

        # NOTE property option (1)
        str: _SupportsGet
        dt: _SupportsGet

        # NOTE property option (2)
        @property
        def cat(self) -> Any: ...
        @property
        def list(self) -> Any: ...

    class DummyExpr(DummyCompliant):
        def __init__(self) -> None:
            self._implementation = nw.Implementation.POLARS

        def alias(self, name: str) -> str:
            return name

        unique = not_implemented()

        # NOTE: Only `mypy` has an issue with this?
        # error: Cannot override writeable attribute with read-only property
        @property
        def str(self) -> PolarsExprStringNamespace:  # type: ignore[override]
            pl_expr = cast("PolarsExpr", self)
            return PolarsExprStringNamespace(pl_expr)

        dt = not_implemented()

        # NOTE: Typing is happy w/ double property
        @property
        def cat(self) -> PolarsExprStringNamespace:
            pl_expr = cast("PolarsExpr", self)
            return PolarsExprStringNamespace(pl_expr)

        # NOTE: Typing still happy - but it complicates runtime (API completeness) access
        _list = not_implemented("list")

        @property
        def list(self) -> Any:
            return self._list

    expr = DummyExpr()
    # NOTE: Happy path
    assert expr._implementation is nw.Implementation.POLARS
    assert expr.alias("new name") == "new name"
    assert isinstance(expr.str, PolarsExprStringNamespace)
    assert isinstance(expr.cat, PolarsExprStringNamespace)

    # NOTE: not implemented override
    pattern = re.compile(
        r".+unique.+ not implemented.+polars", flags=re.DOTALL | re.IGNORECASE
    )
    with pytest.raises(NotImplementedError, match=pattern):
        expr.unique()

    assert isinstance(DummyExpr.unique, not_implemented)
    assert repr(DummyExpr.unique) == "<not_implemented>: DummyExpr.unique"

    pattern = re.compile(
        r".+unique.+ not implemented.+DummyExpr", flags=re.DOTALL | re.IGNORECASE
    )
    with pytest.raises(NotImplementedError, match=pattern):
        DummyExpr.unique()

    pattern = re.compile(
        r".+dt.+ not implemented.+polars", flags=re.DOTALL | re.IGNORECASE
    )
    with pytest.raises(NotImplementedError, match=pattern):
        expr.dt  # noqa: B018

    assert isinstance(DummyExpr.dt, not_implemented)
    assert repr(DummyExpr.dt) == "<not_implemented>: DummyExpr.dt"

    pattern = re.compile(
        r".+list.+ not implemented.+polars", flags=re.DOTALL | re.IGNORECASE
    )
    with pytest.raises(NotImplementedError, match=pattern):
        expr.list  # noqa: B018

    assert isinstance(DummyExpr._list, not_implemented)
    assert repr(DummyExpr._list) == "<not_implemented>: DummyExpr.list"


def test_deprecate_native_namespace() -> None:
    pytest.importorskip("polars")
    import polars as pl

    @deprecate_native_namespace()
    def func1(
        arg: str,  # noqa: ARG001
        *,
        backend: ModuleType | Implementation | str | None = None,  # noqa: ARG001
        native_namespace: ModuleType | None = None,
    ) -> Any:
        return native_namespace

    @deprecate_native_namespace(warn_version="3.0.0")
    def func2(
        arg: str,  # noqa: ARG001
        *,
        backend: ModuleType | Implementation | str | None = None,
        native_namespace: ModuleType | None = None,  # noqa: ARG001
    ) -> Any:
        return backend

    @deprecate_native_namespace(required=True)
    def func3(
        arg: str,  # noqa: ARG001
        *,
        backend: ModuleType | Implementation | str | None = None,
        native_namespace: ModuleType | None = None,  # noqa: ARG001
    ) -> Any:
        return backend

    param = "hello"
    non_default = cast("ModuleType", "non_default")

    assert func1(param, native_namespace=non_default) is None
    with pytest.warns(
        DeprecationWarning,
        match="`native_namespace` is deprecated, please use `backend` instead",
    ):
        result = func2(param, native_namespace=pl)
    assert result is pl
    assert func2(param, backend=pl) is pl

    with pytest.raises(ValueError, match=r"`backend` must be specified in `func3`"):
        func3(param)

    with pytest.raises(
        ValueError, match=r"Can't pass both `native_namespace` and `backend`"
    ):
        func3(param, backend=pl, native_namespace=pl)

    assert func3(param, backend=Implementation.POLARS) is Implementation.POLARS


def test_requires() -> None:
    class ProbablyCompliant:
        _implementation: Implementation = Implementation.POLARS
        _version: Version = Version.MAIN

        def __init__(self, native_obj: str, backend_version: tuple[int, ...]) -> None:
            self._native_obj: str = native_obj
            self._backend_version: tuple[int, ...] = backend_version

        @property
        def native(self) -> str:
            return self._native_obj

        @requires.backend_version((1, 0, 0))
        def to_int(self) -> int:
            return int(self.native)

        @requires.backend_version((2,), hint="Something helpful I suppose")
        def concat(self, *strings: str, separator: str = "") -> str:
            return separator.join((self.native, *strings))

        @requires.backend_version((3, 0, 0))
        def repeat(self, n: int) -> str:
            return self.native * n

    v_05 = ProbablyCompliant("123", (0, 5))
    v_201 = ProbablyCompliant("123", (2, 0, 1))
    v_300 = ProbablyCompliant("123", (3, 0, 0))

    converted = v_201.to_int()
    assert converted == 123
    match = r"`to_int`.+\'polars>=1.0.0\'.+found.+\'0.5\'"
    with pytest.raises(NotImplementedError, match=match):
        v_05.to_int()
    repeated = v_300.repeat(3)
    assert repeated == "123123123"
    match = r"`repeat`.+\'polars>=3.0.0\'.+found.+\'2.0.1\'"
    with pytest.raises(NotImplementedError, match=match):
        v_201.repeat(3)
    match = r"`repeat`.+\'polars>=3.0.0\'.+found.+\'0.5\'"
    with pytest.raises(NotImplementedError, match=match):
        v_05.repeat(3)
    joined = v_201.concat("456", "789")
    assert joined == "123456789"
    joined_sep = v_201.concat("456", "789", separator=" ")
    assert joined_sep == "123 456 789"
    assert v_300.concat("forever") == "123forever"
    pattern = re.compile(
        r"`concat`.+\'polars>=2\'.+found.+\'0.5\'.+Something helpful I suppose", re.DOTALL
    )
    with pytest.raises(NotImplementedError, match=pattern):
        v_05.concat("never")


def test_deferred_iterable() -> None:
    def to_upper(it: Iterable[str]) -> Callable[[], Iterator[str]]:
        def fn() -> Iterator[str]:
            for el in it:
                yield el.capitalize()

        return fn

    iterable = list("hello")
    deferred_1 = _DeferredIterable(iterable.copy)
    deferred_2 = _DeferredIterable(to_upper(iterable))

    assert deferred_1.to_tuple() == tuple("hello")
    assert next(iter(deferred_1)) == "h"
    assert list(deferred_1) == list("hello")
    assert "".join(chain(deferred_1, deferred_2)) == "helloHELLO"
