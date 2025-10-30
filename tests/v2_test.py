# Test assorted functions which we overwrite in stable.v2

from __future__ import annotations

import re
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

import narwhals as nw
import narwhals.stable.v2 as nw_v2
from narwhals._utils import Implementation
from narwhals.exceptions import ShapeError
from narwhals.utils import Version
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    assert_equal_series,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import assert_type

    from narwhals._typing import EagerAllowed
    from narwhals.stable.v2.typing import IntoDataFrameT
    from narwhals.typing import IntoDType, _1DArray, _2DArray


def test_toplevel() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw_v2.from_native(
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, None, 9]})
    )
    result = df.select(
        min=nw_v2.min("a"),
        max=nw_v2.max("a"),
        mean=nw_v2.mean("a"),
        median=nw_v2.median("a"),
        sum=nw_v2.sum("a"),
        sum_h=nw_v2.sum_horizontal("a"),
        min_h=nw_v2.min_horizontal("a"),
        max_h=nw_v2.max_horizontal("a"),
        mean_h=nw_v2.mean_horizontal("a"),
        len=nw_v2.len(),
        concat_str=nw_v2.concat_str(nw_v2.lit("a"), nw_v2.lit("b")),
        fmt=nw_v2.format("{}", "a"),
        any_h=nw_v2.any_horizontal(nw_v2.lit(True), nw_v2.lit(True), ignore_nulls=True),
        all_h=nw_v2.all_horizontal(nw_v2.lit(True), nw_v2.lit(True), ignore_nulls=True),
        first=nw_v2.nth(0),
        no_first=nw_v2.exclude("a", "c"),
        coalesce=nw_v2.coalesce("c", "a"),
    )
    expected = {
        "min": [1, 1, 1],
        "max": [3, 3, 3],
        "mean": [2.0, 2.0, 2.0],
        "median": [2.0, 2.0, 2.0],
        "sum": [6, 6, 6],
        "sum_h": [1, 2, 3],
        "min_h": [1, 2, 3],
        "max_h": [1, 2, 3],
        "mean_h": [1, 2, 3],
        "len": [3, 3, 3],
        "concat_str": ["ab", "ab", "ab"],
        "fmt": ["1", "2", "3"],
        "any_h": [True, True, True],
        "all_h": [True, True, True],
        "first": [1, 2, 3],
        "no_first": [4, 5, 6],
        "coalesce": [7, 2, 9],
    }
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.DataFrame)


def test_when_then() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw_v2.from_native(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [6, 7, 8]}))
    result = df.select(nw_v2.when(nw_v2.col("a") > 1).then("b").otherwise("c"))
    expected = {"b": [6, 5, 6]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.DataFrame)


def test_constructors() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import numpy as np
    import pandas as pd

    if PANDAS_VERSION < (2, 2):
        pytest.skip()
    assert nw_v2.new_series("a", [1, 2, 3], backend="pandas").to_list() == [1, 2, 3]
    arr: np.ndarray[tuple[int, int], Any] = np.array([[1, 2], [3, 4]])  # pyright: ignore[reportAssignmentType]
    result = nw_v2.from_numpy(arr, schema=["a", "b"], backend="pandas")
    assert_equal_data(result, {"a": [1, 3], "b": [2, 4]})
    assert isinstance(result, nw_v2.DataFrame)
    result = nw_v2.from_numpy(
        arr,
        schema=nw_v2.Schema({"a": nw_v2.Int64(), "b": nw_v2.Int64()}),
        backend="pandas",
    )
    assert_equal_data(result, {"a": [1, 3], "b": [2, 4]})
    assert isinstance(result, nw_v2.DataFrame)
    result = nw_v2.from_dict({"a": [1, 2, 3]}, backend="pandas")
    assert_equal_data(result, {"a": [1, 2, 3]})
    assert isinstance(result, nw_v2.DataFrame)
    result = nw_v2.from_arrow(pd.DataFrame({"a": [1, 2, 3]}), backend="pandas")
    assert_equal_data(result, {"a": [1, 2, 3]})
    assert isinstance(result, nw_v2.DataFrame)
    assert isinstance(result, nw_v2.DataFrame)
    result = nw_v2.from_dicts([{"a": 1}, {"a": 2}, {"a": 3}], backend="pandas")
    assert_equal_data(result, {"a": [1, 2, 3]})
    assert isinstance(result, nw_v2.DataFrame)


def test_join() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw_v2.from_native(pd.DataFrame({"a": [1, 2, 3]})).lazy()
    result = df.join(df, how="inner", on="a").sort("a")
    expected = {"a": [1, 2, 3]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.LazyFrame)
    result_eager = df.collect().join(df.collect(), how="inner", on="a")
    assert_equal_data(result_eager, expected)
    assert isinstance(result_eager, nw_v2.DataFrame)


def test_values_counts_v2() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw_v2.from_native(pd.DataFrame({"a": [1, 2, 3]}), eager_only=True)
    result = df["a"].value_counts().sort("a")
    expected = {"a": [1, 2, 3], "count": [1, 1, 1]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.DataFrame)


def test_to_frame() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw_v2.from_native(pd.DataFrame({"a": [1, 2, 3]}), eager_only=True)
    s = df["a"]
    assert isinstance(s, nw_v2.Series)
    df = s.to_frame()
    assert isinstance(df, nw_v2.DataFrame)


def test_is_duplicated_unique() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw_v2.from_native(pd.DataFrame({"a": [1, 2, 3]}), eager_only=True)
    assert df.is_duplicated().to_list() == [False, False, False]
    assert df.is_unique().to_list() == [True, True, True]
    assert isinstance(df.is_duplicated(), nw_v2.Series)
    assert isinstance(df.is_unique(), nw_v2.Series)


def test_concat() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    df = nw_v2.from_native(pa.table({"a": [1, 2, 3]}), eager_only=True)
    result = nw_v2.concat([df, df], how="vertical")
    expected = {"a": [1, 2, 3, 1, 2, 3]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.DataFrame)
    if TYPE_CHECKING:
        assert_type(result, nw_v2.DataFrame[Any])


def test_to_dict_as_series() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    df = nw_v2.from_native(pa.table({"a": [1, 2, 3]}), eager_only=True)
    result = df.to_dict(as_series=True)
    expected = {"a": [1, 2, 3]}
    assert_equal_data(result, expected)
    assert isinstance(result["a"], nw_v2.Series)


def test_from_native_already_nw() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw_v2.from_native(pl.DataFrame({"a": [1]}))
    assert isinstance(nw_v2.from_native(df), nw_v2.DataFrame)
    assert nw_v2.from_native(df) is df
    lf = nw_v2.from_native(pl.LazyFrame({"a": [1]}))
    assert isinstance(nw_v2.from_native(lf), nw_v2.LazyFrame)
    assert nw_v2.from_native(lf) is lf
    s = df["a"]
    assert isinstance(nw_v2.from_native(s, series_only=True), nw_v2.Series)
    assert nw_v2.from_native(df) is df


def test_from_native_invalid_kwds() -> None:
    pytest.importorskip("polars")
    import polars as pl

    with pytest.raises(TypeError, match="got an unexpected keyword"):
        nw_v2.from_native(pl.DataFrame({"a": [1]}), belugas=True)  # type: ignore[call-overload]


def test_io(tmpdir: pytest.TempdirFactory) -> None:
    pytest.importorskip("polars")
    import polars as pl

    csv_filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    parquet_filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    pl.DataFrame({"a": [1]}).write_csv(csv_filepath)
    pl.DataFrame({"a": [1]}).write_parquet(parquet_filepath)
    assert isinstance(nw_v2.read_csv(csv_filepath, backend="polars"), nw_v2.DataFrame)
    assert isinstance(nw_v2.scan_csv(csv_filepath, backend="polars"), nw_v2.LazyFrame)
    assert isinstance(
        nw_v2.read_parquet(parquet_filepath, backend="polars"), nw_v2.DataFrame
    )
    assert isinstance(
        nw_v2.scan_parquet(parquet_filepath, backend="polars"), nw_v2.LazyFrame
    )


def test_narwhalify() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    data = {"a": [2, 3, 4]}

    @nw_v2.narwhalify
    def func(df: nw_v2.DataFrame[IntoDataFrameT]) -> nw_v2.DataFrame[IntoDataFrameT]:
        return df.with_columns(nw_v2.all() + 1)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    data = {"a": [2, 3, 4]}

    class Foo:
        @nw_v2.narwhalify
        def func(
            self, df: nw_v2.DataFrame[IntoDataFrameT], a: int = 1
        ) -> nw_v2.DataFrame[IntoDataFrameT]:
            return df.with_columns(nw_v2.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method_called() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    data = {"a": [2, 3, 4]}

    class Foo:
        @nw_v2.narwhalify
        def func(
            self, df: nw_v2.DataFrame[IntoDataFrameT], a: int = 1
        ) -> nw_v2.DataFrame[IntoDataFrameT]:
            return df.with_columns(nw_v2.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_backends_cross() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")
    import pandas as pd
    import polars as pl

    data = {"a": [2, 3, 4]}

    @nw_v2.narwhalify
    def func(
        arg1: Any, arg2: Any, extra: int = 1
    ) -> tuple[Any, Any, int]:  # pragma: no cover
        return arg1, arg2, extra

    msg = re.escape(
        "Found multiple backends. Make sure that all dataframe/series inputs come from the same backend."
    )
    with pytest.raises(ValueError, match=msg):
        func(pd.DataFrame(data), pl.DataFrame(data))


def test_narwhalify_method_invalid() -> None:
    class Foo:
        @nw_v2.narwhalify(pass_through=False, eager_only=True)
        def func(self) -> Foo:  # pragma: no cover
            return self

        @nw_v2.narwhalify(pass_through=False, eager_only=True)
        def fun2(self, df: Any) -> Any:  # pragma: no cover
            return df

    with pytest.raises(TypeError):
        Foo().func()


def test_with_version(constructor: Constructor) -> None:
    lf = nw_v2.from_native(constructor({"a": [1, 2]})).lazy()
    assert isinstance(lf, nw_v2.LazyFrame)
    assert lf._compliant_frame._with_version(Version.MAIN)._version is Version.MAIN


def test_get_column() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    def minimal_function(data: nw_v2.Series[Any]) -> None:
        data.is_null()

    pd_df = pd.DataFrame({"col": [1, 2, None, 4]})
    col = nw_v2.from_native(pd_df, eager_only=True).get_column("col")
    # check this doesn't raise type-checking errors
    minimal_function(col)
    assert isinstance(col, nw_v2.Series)


def test_imports() -> None:
    # check these don't raise
    from narwhals.stable.v2.dependencies import is_pandas_dataframe  # noqa: F401
    from narwhals.stable.v2.dtypes import Enum  # noqa: F401
    from narwhals.stable.v2.selectors import datetime  # noqa: F401
    from narwhals.stable.v2.typing import IntoDataFrame  # noqa: F401


def test_dataframe_from_dict(eager_backend: EagerAllowed) -> None:
    schema = {"c": nw_v2.Int16(), "d": nw_v2.Float32()}
    result = nw_v2.DataFrame.from_dict(
        {"c": [1, 2], "d": [5, 6]}, backend=eager_backend, schema=schema
    )
    assert result.collect_schema() == schema
    assert result._version is Version.V2
    assert isinstance(result, nw_v2.DataFrame)


def test_dataframe_from_dicts(eager_backend: EagerAllowed) -> None:
    schema = {"c": nw_v2.Int16(), "d": nw_v2.Float32()}
    result = nw_v2.DataFrame.from_dicts(
        [{"c": 1, "d": 5}, {"c": 2, "d": 6}], backend=eager_backend, schema=schema
    )
    assert result.collect_schema() == schema
    assert result._version is Version.V2
    assert isinstance(result, nw_v2.DataFrame)


def test_dataframe_from_arrow(eager_backend: EagerAllowed) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    is_pyarrow = eager_backend in {Implementation.PYARROW, "pyarrow"}
    data: dict[str, Any] = {"ab": [1, 2, 3], "ba": ["four", "five", None]}
    table = pa.table(data)
    supports_arrow_c_stream = nw_v2.DataFrame.from_arrow(table, backend=eager_backend)
    assert_equal_data(supports_arrow_c_stream, data)
    assert isinstance(supports_arrow_c_stream, nw_v2.DataFrame)
    assert supports_arrow_c_stream._version is Version.V2
    if is_pyarrow:
        assert isinstance(supports_arrow_c_stream.to_native(), pa.Table)
    else:
        assert not isinstance(supports_arrow_c_stream.to_native(), pa.Table)
    if PYARROW_VERSION < (14,):  # pragma: no cover
        ...
    else:
        result = nw_v2.DataFrame.from_arrow(
            supports_arrow_c_stream, backend=eager_backend
        )
        assert_equal_data(result, data)
        assert result._version is Version.V2
        assert isinstance(result, nw_v2.DataFrame)
        if is_pyarrow:
            assert isinstance(result.to_native(), pa.Table)
        else:
            assert not isinstance(result.to_native(), pa.Table)


def test_dataframe_from_numpy(eager_backend: EagerAllowed) -> None:
    pytest.importorskip("numpy")
    import numpy as np

    arr: _2DArray = cast("_2DArray", np.array([[5, 2, 0, 1], [1, 4, 7, 8], [1, 2, 3, 9]]))
    schema = {
        "c": nw_v2.Int16(),
        "d": nw_v2.Float32(),
        "e": nw_v2.Int16(),
        "f": nw_v2.Float64(),
    }
    expected = {"c": [5, 1, 1], "d": [2, 4, 2], "e": [0, 7, 3], "f": [1, 8, 9]}
    result = nw_v2.DataFrame.from_numpy(arr, backend=eager_backend, schema=schema)
    result_schema = result.collect_schema()
    assert result._version is Version.V2
    assert isinstance(result, nw_v2.DataFrame)
    assert result_schema == schema
    assert_equal_data(result, expected)

    # NOTE: Existing bug, `schema` and `collect_schema` should be redefined for `v2`
    with pytest.raises(AssertionError):
        assert isinstance(result_schema, nw_v2.Schema)

    assert isinstance(result_schema, nw.Schema)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (None, [5, 2, 0, 1]),
        (nw_v2.Int64, [5, 2, 0, 1]),
        (nw_v2.Int16(), [5, 2, 0, 1]),
        (nw_v2.Float64, [5.0, 2.0, 0.0, 1.0]),
        (nw_v2.Float32(), [5.0, 2.0, 0.0, 1.0]),
    ],
    ids=str,
)
def test_series_from_numpy(
    eager_backend: EagerAllowed, dtype: IntoDType | None, expected: Sequence[Any]
) -> None:
    pytest.importorskip("numpy")
    import numpy as np

    arr: _1DArray = cast("_1DArray", np.array([5, 2, 0, 1]))
    name = "abc"
    result = nw_v2.Series.from_numpy(name, arr, backend=eager_backend, dtype=dtype)
    assert result._version is Version.V2
    assert isinstance(result, nw_v2.Series)
    if dtype:
        assert result.dtype == dtype
    assert_equal_data(result.to_frame(), {name: expected})


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (None, [5, 2, 0, 1]),
        (nw_v2.Int64, [5, 2, 0, 1]),
        (nw_v2.String, ("a", "b", "c")),
        (nw_v2.Float64, [5.0, 2.0, 0.0, 1.0]),
        (nw_v2.Datetime, deque([datetime(2005, 1, 1, 10), datetime(2002, 1, 1, 10, 43)])),
    ],
    ids=str,
)
def test_series_from_iterable(
    eager_backend: EagerAllowed, dtype: IntoDType | None, expected: Sequence[Any]
) -> None:
    data = expected
    name = "abc"
    result = nw_v2.Series.from_iterable(name, data, backend=eager_backend, dtype=dtype)
    assert result._version is Version.V2
    assert isinstance(result, nw_v2.Series)
    if dtype:
        assert result.dtype == dtype
    assert_equal_series(result, expected, name)


def test_mode_single_expr(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 1, 2, 2, 3], "b": [1, 2, 3, 3, 4]}
    df = nw_v2.from_native(constructor_eager(data))
    result = df.select(nw_v2.col("a").mode()).sort("a")
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)


def test_mode_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 1, 2, 2, 3], "b": [1, 2, 3, 3, 4]}
    series = nw_v2.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.mode().sort()
    expected = {"a": [1, 2]}
    assert_equal_data({"a": result}, expected)


def test_mode_different_lengths(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 10):
        pytest.skip()
    df = nw_v2.from_native(constructor_eager({"a": [1, 1, 2], "b": [4, 5, 6]}))
    with pytest.raises(ShapeError):
        df.select(nw_v2.col("a", "b").mode())
