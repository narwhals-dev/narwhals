from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.functions import _get_deps_info
from narwhals.functions import _get_sys_info
from narwhals.functions import show_versions
from narwhals.utils import parse_version
from tests.utils import compare_dicts
from tests.utils import maybe_get_modin_df

if TYPE_CHECKING:
    from narwhals.typing import IntoFrameT

df_pandas = pd.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
if parse_version(pd.__version__) >= parse_version("1.5.0"):
    df_pandas_pyarrow = pd.DataFrame(
        {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    ).astype(
        {
            "a": "Int64[pyarrow]",
            "b": "Int64[pyarrow]",
            "z": "Float64[pyarrow]",
        }
    )
    df_pandas_nullable = pd.DataFrame(
        {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    ).astype(
        {
            "a": "Int64",
            "b": "Int64",
            "z": "Float64",
        }
    )
else:  # pragma: no cover
    df_pandas_pyarrow = df_pandas
    df_pandas_nullable = df_pandas
df_polars = pl.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
df_lazy = pl.LazyFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
df_pandas_na = pd.DataFrame({"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]})
df_lazy_na = pl.LazyFrame({"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]})
df_right_pandas = pd.DataFrame({"c": [6, 12, -1], "d": [0, -4, 2]})
df_right_lazy = pl.LazyFrame({"c": [6, 12, -1], "d": [0, -4, 2]})
df_mpd = maybe_get_modin_df(df_pandas)
df_pa = pa.table({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
df_pa_na = pa.table({"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]})


@pytest.mark.parametrize(
    "constructor",
    [pd.DataFrame, pl.DataFrame, pa.table],
)
def test_empty_select(constructor: Any) -> None:
    result = nw.from_native(constructor({"a": [1, 2, 3]}), eager_only=True).select()
    assert result.shape == (0, 0)


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow],
)
def test_std(df_raw: Any) -> None:
    df = nw.from_native(df_raw)
    result = df.select(
        nw.col("a").std().alias("a_ddof_default"),
        nw.col("a").std(ddof=1).alias("a_ddof_1"),
        nw.col("a").std(ddof=0).alias("a_ddof_0"),
        nw.col("b").std(ddof=2).alias("b_ddof_2"),
        nw.col("z").std(ddof=0).alias("z_ddof_0"),
    )
    result_native = nw.to_native(result)
    expected = {
        "a_ddof_default": [1.0],
        "a_ddof_1": [1.0],
        "a_ddof_0": [0.816497],
        "b_ddof_2": [1.632993],
        "z_ddof_0": [0.816497],
    }
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
def test_schema(df_raw: Any) -> None:
    result = nw.from_native(df_raw).schema
    expected = {"a": nw.Int64, "b": nw.Int64, "z": nw.Float64}
    assert result == expected
    result = nw.from_native(df_raw).lazy().collect().schema
    expected = {"a": nw.Int64, "b": nw.Int64, "z": nw.Float64}
    assert result == expected
    result = nw.from_native(df_raw).columns  # type: ignore[assignment]
    expected = ["a", "b", "z"]  # type: ignore[assignment]
    assert result == expected
    result = nw.from_native(df_raw).lazy().collect().columns  # type: ignore[assignment]
    expected = ["a", "b", "z"]  # type: ignore[assignment]
    assert result == expected


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
def test_columns(df_raw: Any) -> None:
    df = nw.from_native(df_raw)
    result = df.columns
    expected = ["a", "b", "z"]
    assert result == expected


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_mpd, df_lazy])
def test_expr_binary(df_raw: Any) -> None:
    result = nw.from_native(df_raw).with_columns(
        a=(1 + 3 * nw.col("a")) * (1 / nw.col("a")),
        b=nw.col("z") / (2 - nw.col("b")),
        c=nw.col("a") + nw.col("b") / 2,
        d=nw.col("a") - nw.col("b"),
        e=((nw.col("a") > nw.col("b")) & (nw.col("a") >= nw.col("z"))).cast(nw.Int64),
        f=(
            (nw.col("a") < nw.col("b"))
            | (nw.col("a") <= nw.col("z"))
            | (nw.col("a") == 1)
        ).cast(nw.Int64),
        g=nw.col("a") != 1,
        h=(False & (nw.col("a") != 1)),
        i=(False | (nw.col("a") != 1)),
        j=2 ** nw.col("a"),
        k=2 // nw.col("a"),
        l=nw.col("a") // 2,
        m=nw.col("a") ** 2,
        n=nw.col("a") % 2,
        o=2 % nw.col("a"),
    )
    result_native = nw.to_native(result)
    expected = {
        "a": [4, 3.333333, 3.5],
        "b": [-3.5, -4.0, -2.25],
        "z": [7.0, 8.0, 9.0],
        "c": [3, 5, 5],
        "d": [-3, -1, -4],
        "e": [0, 0, 0],
        "f": [1, 1, 1],
        "g": [False, True, True],
        "h": [False, False, False],
        "i": [False, True, True],
        "j": [2, 8, 4],
        "k": [2, 0, 1],
        "l": [0, 1, 1],
        "m": [1, 9, 4],
        "n": [1, 1, 0],
        "o": [0, 2, 0],
    }
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_mpd, df_lazy])
def test_expr_transform(df_raw: Any) -> None:
    result = nw.from_native(df_raw).with_columns(
        a=nw.col("a").is_between(-1, 1), b=nw.col("b").is_in([4, 5])
    )
    result_native = nw.to_native(result)
    expected = {"a": [True, False, False], "b": [True, True, False], "z": [7, 8, 9]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_lazy])
def test_expr_min_max(df_raw: Any) -> None:
    df = nw.from_native(df_raw)
    result_min = nw.to_native(df.select(nw.min("a", "b", "z")))
    result_max = nw.to_native(df.select(nw.max("a", "b", "z")))
    expected_min = {"a": [1], "b": [4], "z": [7]}
    expected_max = {"a": [3], "b": [6], "z": [9]}
    compare_dicts(result_min, expected_min)
    compare_dicts(result_max, expected_max)


@pytest.mark.parametrize("df_raw", [df_pandas_na, df_lazy_na])
def test_expr_na(df_raw: Any) -> None:
    df = nw.from_native(df_raw).lazy()
    result_nna = nw.to_native(
        df.filter((~nw.col("a").is_null()) & (~df.collect()["z"].is_null()))
    )
    expected = {"a": [2], "b": [6], "z": [9]}
    compare_dicts(result_nna, expected)


@pytest.mark.parametrize("df_raw", [df_pandas_na, df_lazy_na])
def test_drop_nulls(df_raw: Any) -> None:
    df = nw.from_native(df_raw).lazy()
    result = nw.to_native(df.select(nw.col("a").drop_nulls()))
    expected = {"a": [3, 2]}
    compare_dicts(result, expected)
    result = nw.to_native(df.select(df.collect()["a"].drop_nulls()))
    expected = {"a": [3, 2]}
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    "df_raw",
    [
        df_pandas,
        df_polars,
        df_mpd,
        df_pa,
    ],
)
@pytest.mark.parametrize(
    ("drop", "left"),
    [
        (["a"], ["b", "z"]),
        (["a", "b"], ["z"]),
    ],
)
def test_drop(df_raw: Any, drop: list[str], left: list[str]) -> None:
    df = nw.from_native(df_raw)
    assert df.drop(drop).columns == left
    assert df.drop(*drop).columns == left


@pytest.mark.parametrize(
    ("df_raw", "df_raw_right"), [(df_pandas, df_right_pandas), (df_lazy, df_right_lazy)]
)
def test_concat_horizontal(df_raw: IntoFrameT, df_raw_right: IntoFrameT) -> None:
    df_left = nw.from_native(df_raw)
    df_right = nw.from_native(df_raw_right)
    result = nw.concat([df_left, df_right], how="horizontal")
    result_native = nw.to_native(result)
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8, 9],
        "c": [6, 12, -1],
        "d": [0, -4, 2],
    }
    compare_dicts(result_native, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([])


@pytest.mark.parametrize(
    ("df_raw", "df_raw_right"), [(df_pandas, df_right_pandas), (df_lazy, df_right_lazy)]
)
def test_concat_vertical(df_raw: Any, df_raw_right: Any) -> None:
    df_left = nw.from_native(df_raw).rename({"a": "c", "b": "d"}).drop("z").lazy()
    df_right = nw.from_native(df_raw_right).lazy()
    result = nw.concat([df_left, df_right], how="vertical")
    result_native = nw.to_native(result)
    expected = {"c": [1, 3, 2, 6, 12, -1], "d": [4, 4, 6, 0, -4, 2]}
    compare_dicts(result_native, expected)
    with pytest.raises(ValueError, match="No items"):
        nw.concat([], how="vertical")
    with pytest.raises(Exception, match="unable to vstack"):
        nw.concat([df_left, df_right.rename({"d": "i"})], how="vertical").collect()


@pytest.mark.parametrize("df_raw", [df_pandas, df_polars, df_pa])
def test_lazy(df_raw: Any) -> None:
    df = nw.from_native(df_raw, eager_only=True)
    result = df.lazy()
    assert isinstance(result, nw.LazyFrame)


def test_invalid() -> None:
    df = nw.from_native(pa.table({"a": [1, 2], "b": [3, 4]}))
    with pytest.raises(ValueError, match="Multi-output"):
        df.select(nw.all() + nw.all())
    df = nw.from_native(df_pandas)
    with pytest.raises(ValueError, match="Multi-output"):
        df.select(nw.all() + nw.all())
    with pytest.raises(TypeError, match="Perhaps you:"):
        df.select([pl.col("a")])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="Perhaps you:"):
        df.select([nw.col("a").cast(pl.Int64)])


@pytest.mark.parametrize("df_raw", [df_pandas])
def test_reindex(df_raw: Any) -> None:
    df = nw.from_native(df_raw, eager_only=True)
    result = df.select("b", df["a"].sort(descending=True))
    expected = {"b": [4, 4, 6], "a": [3, 2, 1]}
    compare_dicts(result, expected)
    result = df.select("b", nw.col("a").sort(descending=True))
    compare_dicts(result, expected)

    s = df["a"]
    result_s = s > s.sort()
    assert not result_s[0]
    assert result_s[1]
    assert not result_s[2]
    result = df.with_columns(s.sort())
    expected = {"a": [1, 2, 3], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}  # type: ignore[list-item]
    compare_dicts(result, expected)
    with pytest.raises(ValueError, match="Multi-output expressions are not supported"):
        nw.to_native(df.with_columns(nw.all() + nw.all()))


@pytest.mark.parametrize(
    ("df_raw", "df_raw_right"),
    [(df_pandas, df_polars), (df_polars, df_pandas)],
)
def test_library(df_raw: Any, df_raw_right: Any) -> None:
    df_left = nw.from_native(df_raw).lazy()
    df_right = nw.from_native(df_raw_right).lazy()
    with pytest.raises(
        NotImplementedError, match="Cross-library comparisons aren't supported"
    ):
        nw.concat([df_left, df_right], how="horizontal")
    with pytest.raises(
        NotImplementedError, match="Cross-library comparisons aren't supported"
    ):
        nw.concat([df_left, df_right], how="vertical")
    with pytest.raises(
        NotImplementedError, match="Cross-library comparisons aren't supported"
    ):
        df_left.join(df_right, left_on=["a"], right_on=["a"], how="inner")


@pytest.mark.parametrize("df_raw", [df_pandas, df_polars])
@pytest.mark.parametrize(
    ("row", "column", "expected"),
    [(0, 2, 7), (1, "z", 8)],
)
def test_item(
    df_raw: Any,
    row: int | None,
    column: int | str | None,
    expected: Any,
) -> None:
    df = nw.from_native(df_raw, eager_only=True)
    assert df.item(row, column) == expected
    assert df.select("a").head(1).item() == 1


@pytest.mark.parametrize("df_raw", [df_pandas, df_polars])
@pytest.mark.parametrize(
    ("row", "column", "err_msg"),
    [
        (0, None, re.escape("cannot call `.item()` with only one of `row` or `column`")),
        (None, 0, re.escape("cannot call `.item()` with only one of `row` or `column`")),
        (
            None,
            None,
            re.escape("can only call `.item()` if the dataframe is of shape (1, 1)"),
        ),
    ],
)
def test_item_value_error(
    df_raw: Any,
    row: int | None,
    column: int | str | None,
    err_msg: str,
) -> None:
    with pytest.raises(ValueError, match=err_msg):
        nw.from_native(df_raw, eager_only=True).item(row, column)


@pytest.mark.parametrize("df_raw", [df_pandas, df_polars])
def test_with_columns_order(df_raw: Any) -> None:
    df = nw.from_native(df_raw)
    result = df.with_columns(nw.col("a") + 1, d=nw.col("a") - 1)
    assert result.columns == ["a", "b", "z", "d"]
    expected = {"a": [2, 4, 3], "b": [4, 4, 6], "z": [7.0, 8, 9], "d": [0, 2, 1]}
    compare_dicts(result, expected)


@pytest.mark.parametrize("df_raw", [df_pandas, df_polars])
def test_with_columns_order_single_row(df_raw: Any) -> None:
    df = nw.from_native(df_raw[:1])
    assert len(df) == 1
    result = df.with_columns(nw.col("a") + 1, d=nw.col("a") - 1)
    assert result.columns == ["a", "b", "z", "d"]
    expected = {"a": [2], "b": [4], "z": [7.0], "d": [0]}
    compare_dicts(result, expected)


def test_get_sys_info() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        show_versions()
        sys_info = _get_sys_info()

    assert "python" in sys_info
    assert "executable" in sys_info
    assert "machine" in sys_info


def test_get_deps_info() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        show_versions()
        deps_info = _get_deps_info()

    assert "narwhals" in deps_info
    assert "pandas" in deps_info
    assert "polars" in deps_info
    assert "cudf" in deps_info
    assert "modin" in deps_info
    assert "pyarrow" in deps_info
    assert "numpy" in deps_info


def test_show_versions(capsys: Any) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        show_versions()
        out, err = capsys.readouterr()

    assert "python" in out
    assert "machine" in out
    assert "pandas" in out
    assert "polars" in out
