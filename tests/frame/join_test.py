from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    assert_equal_data,
)


@pytest.mark.parametrize(
    ("df1", "df2", "expected", "on", "left_on", "right_on"),
    [
        (
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]},
            {
                "id": [2, 3, 4],
                "department": ["HR", "Engineering", "Marketing"],
                "salary": [50000, 60000, 70000],
            },
            {
                "id": [1, 2, 3, None],
                "name": ["Alice", "Bob", "Charlie", None],
                "age": [25, 30, 35, None],
                "id_right": [None, 2, 3, 4],
                "department": [None, "HR", "Engineering", "Marketing"],
                "salary": [None, 50000, 60000, 70000],
            },
            None,
            ["id"],
            ["id"],
        ),
        (
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]},
            {
                "id": [2, 3, 4],
                "department": ["HR", "Engineering", "Marketing"],
                "salary": [50000, 60000, 70000],
            },
            {
                "id": [1, 2, 3, None],
                "name": ["Alice", "Bob", "Charlie", None],
                "age": [25, 30, 35, None],
                "id_right": [None, 2, 3, 4],
                "department": [None, "HR", "Engineering", "Marketing"],
                "salary": [None, 50000, 60000, 70000],
            },
            "id",
            None,
            None,
        ),
        (
            {
                "id": [1, 2, 3, 4],
                "year": [2020, 2021, 2022, 2023],
                "value1": [100, 200, 300, 400],
            },
            {
                "id": [2, 3, 4, 5],
                "year_foo": [2021, 2022, 2023, 2024],
                "value2": [500, 600, 700, 800],
            },
            {
                "id": [1, 2, 3, 4, None],
                "year": [2020, 2021, 2022, 2023, None],
                "value1": [100, 200, 300, 400, None],
                "id_right": [None, 2, 3, 4, 5],
                # since year is different, don't apply suffix
                "year_foo": [None, 2021, 2022, 2023, 2024],
                "value2": [None, 500, 600, 700, 800],
            },
            None,
            ["id", "year"],
            ["id", "year_foo"],
        ),
    ],
)
def test_full_join(
    df1: dict[str, list[Any]],
    df2: dict[str, list[Any]],
    expected: dict[str, list[Any]],
    on: None | str | list[str],
    left_on: None | str | list[str],
    right_on: None | str | list[str],
    constructor: Constructor,
) -> None:
    df_left = nw.from_native(constructor(df1))
    df_right = nw.from_native(constructor(df2))
    result = df_left.join(
        df_right, on=on, left_on=left_on, right_on=right_on, how="full"
    ).sort("id", nulls_last=True)
    assert_equal_data(result, expected)


def test_full_join_duplicate(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "ibis" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df1 = {"foo": [1, 2, 3], "val1": [1, 2, 3]}
    df2 = {"foo": [1, 2, 3], "foo_right": [1, 2, 3]}
    df_left = nw.from_native(constructor(df1)).lazy()
    df_right = nw.from_native(constructor(df2)).lazy()

    exceptions: list[type[Exception]] = [nw.exceptions.NarwhalsError]
    if "pyspark" in str(constructor) and "sqlframe" not in str(constructor):
        from pyspark.errors import AnalysisException

        exceptions.append(AnalysisException)
    elif "cudf" in str(constructor):
        # cudf throw their own exception earlier in the stack
        exceptions.append(ValueError)

    with pytest.raises(tuple(exceptions)):
        df_left.join(df_right, on="foo", how="full").collect()


def test_inner_join_two_keys(constructor: Constructor) -> None:
    data = {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zor ro": [7.0, 8.0, 9.0],
        "idx": [0, 1, 2],
    }
    df = nw.from_native(constructor(data))
    df_right = df
    result = df.join(
        df_right,
        left_on=["antananarivo", "bob"],
        right_on=["antananarivo", "bob"],
        how="inner",
    )
    result_on = df.join(df_right, on=["antananarivo", "bob"], how="inner")
    result = result.sort("idx").drop("idx_right")
    result_on = result_on.sort("idx").drop("idx_right")
    expected = {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zor ro": [7.0, 8.0, 9.0],
        "idx": [0, 1, 2],
        "zor ro_right": [7.0, 8.0, 9.0],
    }
    assert_equal_data(result, expected)
    assert_equal_data(result_on, expected)


def test_inner_join_single_key(constructor: Constructor) -> None:
    data = {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zor ro": [7.0, 8.0, 9.0],
        "idx": [0, 1, 2],
    }
    df = nw.from_native(constructor(data))
    df_right = df
    result = df.join(
        df_right, left_on="antananarivo", right_on="antananarivo", how="inner"
    ).sort("idx")
    result_on = df.join(df_right, on="antananarivo", how="inner").sort("idx")
    result = result.drop("idx_right")
    result_on = result_on.drop("idx_right")
    expected = {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zor ro": [7.0, 8.0, 9.0],
        "idx": [0, 1, 2],
        "bob_right": [4, 4, 6],
        "zor ro_right": [7.0, 8.0, 9.0],
    }
    assert_equal_data(result, expected)
    assert_equal_data(result_on, expected)


def test_cross_join(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 1, 4):
        pytest.skip()
    data = {"antananarivo": [1, 3, 2]}
    df = nw.from_native(constructor(data))
    result = df.join(df, how="cross").sort("antananarivo", "antananarivo_right")
    expected = {
        "antananarivo": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "antananarivo_right": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    }
    assert_equal_data(result, expected)

    with pytest.raises(
        ValueError, match="Can not pass `left_on`, `right_on` or `on` keys for cross join"
    ):
        df.join(df, how="cross", left_on="antananarivo")


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("suffix", ["_right", "_custom_suffix"])
def test_suffix(
    constructor: Constructor, how: Literal["inner", "left"], suffix: str
) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    df_right = df
    result = df.join(
        df_right,
        left_on=["antananarivo", "bob"],
        right_on=["antananarivo", "bob"],
        how=how,
        suffix=suffix,
    )
    result_cols = result.collect_schema().names()
    assert result_cols == ["antananarivo", "bob", "zor ro", f"zor ro{suffix}"]


@pytest.mark.parametrize("suffix", ["_right", "_custom_suffix"])
def test_cross_join_suffix(constructor: Constructor, suffix: str) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 1, 4):
        pytest.skip()
    data = {"antananarivo": [1, 3, 2]}
    df = nw.from_native(constructor(data))
    result = df.join(df, how="cross", suffix=suffix).sort(
        "antananarivo", f"antananarivo{suffix}"
    )
    expected = {
        "antananarivo": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        f"antananarivo{suffix}": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    }
    assert_equal_data(result, expected)


def test_cross_join_non_pandas() -> None:
    _ = pytest.importorskip("modin")

    import modin.pandas as mpd

    data = {"antananarivo": [1, 3, 2]}
    df1 = nw.from_native(mpd.DataFrame(pd.DataFrame(data)), eager_only=True)
    df2 = nw.from_native(mpd.DataFrame(pd.DataFrame(data)), eager_only=True)
    result = df1.join(df2, how="cross")
    expected = {
        "antananarivo": [1, 1, 1, 3, 3, 3, 2, 2, 2],
        "antananarivo_right": [1, 3, 2, 1, 3, 2, 1, 3, 2],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("join_key", "filter_expr", "expected"),
    [
        (
            ["antananarivo", "bob"],
            (nw.col("bob") < 5),
            {"antananarivo": [2], "bob": [6], "zor ro": [9]},
        ),
        (["bob"], (nw.col("bob") < 5), {"antananarivo": [2], "bob": [6], "zor ro": [9]}),
        (
            ["bob"],
            (nw.col("bob") > 5),
            {"antananarivo": [1, 3], "bob": [4, 4], "zor ro": [7.0, 8.0]},
        ),
    ],
)
def test_anti_join(
    constructor: Constructor,
    join_key: list[str],
    filter_expr: nw.Expr,
    expected: dict[str, list[Any]],
) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    other = df.filter(filter_expr)
    result = df.join(other, how="anti", left_on=join_key, right_on=join_key)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("join_key", "filter_expr", "expected"),
    [
        (
            "antananarivo",
            (nw.col("bob") > 5),
            {"antananarivo": [2], "bob": [6], "zor ro": [9]},
        ),
        (
            ["antananarivo"],
            (nw.col("bob") > 5),
            {"antananarivo": [2], "bob": [6], "zor ro": [9]},
        ),
        (
            ["bob"],
            (nw.col("bob") < 5),
            {"antananarivo": [1, 3], "bob": [4, 4], "zor ro": [7, 8]},
        ),
        (
            ["antananarivo", "bob"],
            (nw.col("bob") < 5),
            {"antananarivo": [1, 3], "bob": [4, 4], "zor ro": [7, 8]},
        ),
    ],
)
def test_semi_join(
    constructor: Constructor,
    join_key: list[str],
    filter_expr: nw.Expr,
    expected: dict[str, list[Any]],
) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    other = df.filter(filter_expr)
    result = df.join(other, how="semi", left_on=join_key, right_on=join_key).sort(
        "antananarivo"
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize("how", ["right"])
def test_join_not_implemented(constructor: Constructor, how: str) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            f"Only the following join strategies are supported: ('inner', 'left', 'full', 'cross', 'anti', 'semi'); found '{how}'."
        ),
    ):
        df.join(df, left_on="antananarivo", right_on="antananarivo", how=how)  # type: ignore[arg-type]


def test_left_join(constructor: Constructor) -> None:
    data_left = {
        "antananarivo": [1.0, 2.0, 3.0],
        "bob": [4.0, 5.0, 6.0],
        "idx": [0.0, 1.0, 2.0],
    }
    data_right = {
        "antananarivo": [1.0, 2.0, 3.0],
        "co": [4.0, 5.0, 7.0],
        "idx": [0.0, 1.0, 2.0],
    }
    df_left = nw.from_native(constructor(data_left))
    df_right = nw.from_native(constructor(data_right))
    result = df_left.join(df_right, left_on="bob", right_on="co", how="left")
    result = result.sort("idx")
    result = result.drop("idx_right")
    expected = {
        "antananarivo": [1, 2, 3],
        "bob": [4, 5, 6],
        "idx": [0, 1, 2],
        "antananarivo_right": [1, 2, None],
    }
    result_on_list = df_left.join(df_right, on=["antananarivo", "idx"], how="left")
    result_on_list = result_on_list.sort("idx")
    expected_on_list = {
        "antananarivo": [1, 2, 3],
        "bob": [4, 5, 6],
        "idx": [0, 1, 2],
        "co": [4, 5, 7],
    }
    assert_equal_data(result, expected)
    assert_equal_data(result_on_list, expected_on_list)


def test_left_join_multiple_column(constructor: Constructor) -> None:
    data_left = {"antananarivo": [1, 2, 3], "bob": [4, 5, 6], "idx": [0, 1, 2]}
    data_right = {"antananarivo": [1, 2, 3], "c": [4, 5, 6], "idx": [0, 1, 2]}
    df_left = nw.from_native(constructor(data_left))
    df_right = nw.from_native(constructor(data_right))
    result = df_left.join(
        df_right,
        left_on=["antananarivo", "bob"],
        right_on=["antananarivo", "c"],
        how="left",
    )
    result = result.sort("idx")
    result = result.drop("idx_right")
    expected = {"antananarivo": [1, 2, 3], "bob": [4, 5, 6], "idx": [0, 1, 2]}
    assert_equal_data(result, expected)


def test_left_join_overlapping_column(constructor: Constructor) -> None:
    data_left = {
        "antananarivo": [1.0, 2.0, 3.0],
        "bob": [4.0, 5.0, 6.0],
        "d": [1.0, 4.0, 2.0],
        "idx": [0.0, 1.0, 2.0],
    }
    data_right = {
        "antananarivo": [1.0, 2.0, 3.0],
        "c": [4.0, 5.0, 6.0],
        "d": [1.0, 4.0, 2.0],
        "idx": [0.0, 1.0, 2.0],
    }
    df_left = nw.from_native(constructor(data_left))
    df_right = nw.from_native(constructor(data_right))
    result = df_left.join(df_right, left_on="bob", right_on="c", how="left").sort("idx")
    result = result.drop("idx_right")
    expected: dict[str, list[Any]] = {
        "antananarivo": [1, 2, 3],
        "bob": [4, 5, 6],
        "d": [1, 4, 2],
        "idx": [0, 1, 2],
        "antananarivo_right": [1, 2, 3],
        "d_right": [1, 4, 2],
    }
    assert_equal_data(result, expected)
    result = df_left.join(df_right, left_on="antananarivo", right_on="d", how="left")
    result = result.sort("idx")
    result = result.drop("idx_right")
    expected = {
        "antananarivo": [1, 2, 3],
        "bob": [4, 5, 6],
        "d": [1, 4, 2],
        "idx": [0, 1, 2],
        "antananarivo_right": [1.0, 3.0, None],
        "c": [4.0, 6.0, None],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize("how", ["inner", "left", "semi", "anti"])
def test_join_keys_exceptions(constructor: Constructor, how: str) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))

    with pytest.raises(
        ValueError,
        match=rf"Either \(`left_on` and `right_on`\) or `on` keys should be specified for {how}.",
    ):
        df.join(df, how=how)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=rf"Either \(`left_on` and `right_on`\) or `on` keys should be specified for {how}.",
    ):
        df.join(df, how=how, left_on="antananarivo")  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=rf"Either \(`left_on` and `right_on`\) or `on` keys should be specified for {how}.",
    ):
        df.join(df, how=how, right_on="antananarivo")  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=f"If `on` is specified, `left_on` and `right_on` should be None for {how}.",
    ):
        df.join(df, how=how, on="antananarivo", right_on="antananarivo")  # type: ignore[arg-type]

    with pytest.raises(
        ValueError, match="`left_on` and `right_on` must have the same length."
    ):
        df.join(df, how=how, left_on=["antananarivo", "bob"], right_on="antananarivo")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (
            "backward",
            {"antananarivo": [1, 5, 10], "val": ["a", "b", "c"], "val_right": [1, 3, 7]},
        ),
        (
            "forward",
            {
                "antananarivo": [1, 5, 10],
                "val": ["a", "b", "c"],
                "val_right": [1, 6, None],
            },
        ),
        (
            "nearest",
            {"antananarivo": [1, 5, 10], "val": ["a", "b", "c"], "val_right": [1, 6, 7]},
        ),
    ],
)
def test_joinasof_numeric(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    strategy: Literal["backward", "forward", "nearest"],
    expected: dict[str, list[Any]],
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "cudf", "pyspark")):
        request.applymarker(pytest.mark.xfail)
    if (
        "duckdb" in str(constructor) or "ibis" in str(constructor)
    ) and strategy == "nearest":
        request.applymarker(pytest.mark.xfail)
    if PANDAS_VERSION < (2, 1) and (
        ("pandas_pyarrow" in str(constructor)) or ("pandas_nullable" in str(constructor))
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(
        constructor({"antananarivo": [1, 5, 10], "val": ["a", "b", "c"]})
    ).sort("antananarivo")
    df_right = nw.from_native(
        constructor({"antananarivo": [1, 2, 3, 6, 7], "val": [1, 2, 3, 6, 7]})
    ).sort("antananarivo")
    result = df.join_asof(
        df_right, left_on="antananarivo", right_on="antananarivo", strategy=strategy
    )
    result_on = df.join_asof(df_right, on="antananarivo", strategy=strategy)
    assert_equal_data(result.sort(by="antananarivo"), expected)
    assert_equal_data(result_on.sort(by="antananarivo"), expected)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (
            "backward",
            {
                "datetime": [
                    datetime(2016, 3, 1),
                    datetime(2018, 8, 1),
                    datetime(2019, 1, 1),
                ],
                "population": [82.19, 82.66, 83.12],
                "gdp": [4164, 4566, 4696],
            },
        ),
        (
            "forward",
            {
                "datetime": [
                    datetime(2016, 3, 1),
                    datetime(2018, 8, 1),
                    datetime(2019, 1, 1),
                ],
                "population": [82.19, 82.66, 83.12],
                "gdp": [4411, 4696, 4696],
            },
        ),
        (
            "nearest",
            {
                "datetime": [
                    datetime(2016, 3, 1),
                    datetime(2018, 8, 1),
                    datetime(2019, 1, 1),
                ],
                "population": [82.19, 82.66, 83.12],
                "gdp": [4164, 4696, 4696],
            },
        ),
    ],
)
def test_joinasof_time(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    strategy: Literal["backward", "forward", "nearest"],
    expected: dict[str, list[Any]],
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "cudf", "pyspark")):
        request.applymarker(pytest.mark.xfail)
    if (
        "duckdb" in str(constructor) or "ibis" in str(constructor)
    ) and strategy == "nearest":
        request.applymarker(pytest.mark.xfail)
    if PANDAS_VERSION < (2, 1) and ("pandas_pyarrow" in str(constructor)):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(
        constructor(
            {
                "datetime": [
                    datetime(2016, 3, 1),
                    datetime(2018, 8, 1),
                    datetime(2019, 1, 1),
                ],
                "population": [82.19, 82.66, 83.12],
            }
        )
    ).sort("datetime")
    df_right = nw.from_native(
        constructor(
            {
                "datetime": [
                    datetime(2016, 1, 1),
                    datetime(2017, 1, 1),
                    datetime(2018, 1, 1),
                    datetime(2019, 1, 1),
                    datetime(2020, 1, 1),
                ],
                "gdp": [4164, 4411, 4566, 4696, 4827],
            }
        )
    ).sort("datetime")
    result = df.join_asof(
        df_right, left_on="datetime", right_on="datetime", strategy=strategy
    )
    result_on = df.join_asof(df_right, on="datetime", strategy=strategy)
    assert_equal_data(result.sort(by="datetime"), expected)
    assert_equal_data(result_on.sort(by="datetime"), expected)


def test_joinasof_by(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "cudf", "pyspark")):
        request.applymarker(pytest.mark.xfail)
    if PANDAS_VERSION < (2, 1) and (
        ("pandas_pyarrow" in str(constructor)) or ("pandas_nullable" in str(constructor))
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(
        constructor(
            {
                "antananarivo": [1, 5, 7, 10],
                "bob": ["D", "D", "C", "A"],
                "c": [9, 2, 1, 1],
            }
        )
    ).sort("antananarivo")
    df_right = nw.from_native(
        constructor(
            {"antananarivo": [1, 4, 5, 8], "bob": ["D", "D", "A", "F"], "d": [1, 3, 4, 1]}
        )
    ).sort("antananarivo")
    result = df.join_asof(df_right, on="antananarivo", by_left="bob", by_right="bob")
    result_by = df.join_asof(df_right, on="antananarivo", by="bob")
    expected = {
        "antananarivo": [1, 5, 7, 10],
        "bob": ["D", "D", "C", "A"],
        "c": [9, 2, 1, 1],
        "d": [1, 3, None, 4],
    }
    assert_equal_data(result.sort(by="antananarivo"), expected)
    assert_equal_data(result_by.sort(by="antananarivo"), expected)


def test_joinasof_suffix(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "cudf", "pyspark")):
        request.applymarker(pytest.mark.xfail)
    if PANDAS_VERSION < (2, 1) and (
        ("pandas_pyarrow" in str(constructor)) or ("pandas_nullable" in str(constructor))
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(
        constructor({"antananarivo": [1, 5, 10], "val": ["a", "b", "c"]})
    ).sort("antananarivo")
    df_right = nw.from_native(
        constructor({"antananarivo": [1, 2, 3, 6, 7], "val": [1, 2, 3, 6, 7]})
    ).sort("antananarivo")
    result = df.join_asof(
        df_right, left_on="antananarivo", right_on="antananarivo", suffix="_y"
    )
    expected = {"antananarivo": [1, 5, 10], "val": ["a", "b", "c"], "val_y": [1, 3, 7]}
    assert_equal_data(result.sort(by="antananarivo"), expected)


@pytest.mark.parametrize("strategy", ["back", "furthest"])
def test_joinasof_not_implemented(
    constructor: Constructor, strategy: Literal["backward", "forward"]
) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))

    with pytest.raises(
        NotImplementedError,
        match=rf"Only the following strategies are supported: \('backward', 'forward', 'nearest'\); found '{strategy}'.",
    ):
        df.join_asof(
            df, left_on="antananarivo", right_on="antananarivo", strategy=strategy
        )


def test_joinasof_keys_exceptions(constructor: Constructor) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))

    with pytest.raises(
        ValueError,
        match=r"Either \(`left_on` and `right_on`\) or `on` keys should be specified.",
    ):
        df.join_asof(df, left_on="antananarivo")
    with pytest.raises(
        ValueError,
        match=r"Either \(`left_on` and `right_on`\) or `on` keys should be specified.",
    ):
        df.join_asof(df, right_on="antananarivo")
    with pytest.raises(
        ValueError,
        match=r"Either \(`left_on` and `right_on`\) or `on` keys should be specified.",
    ):
        df.join_asof(df)
    with pytest.raises(
        ValueError, match="If `on` is specified, `left_on` and `right_on` should be None."
    ):
        df.join_asof(
            df, left_on="antananarivo", right_on="antananarivo", on="antananarivo"
        )
    with pytest.raises(
        ValueError, match="If `on` is specified, `left_on` and `right_on` should be None."
    ):
        df.join_asof(df, left_on="antananarivo", on="antananarivo")
    with pytest.raises(
        ValueError, match="If `on` is specified, `left_on` and `right_on` should be None."
    ):
        df.join_asof(df, right_on="antananarivo", on="antananarivo")


def test_joinasof_by_exceptions(constructor: Constructor) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    with pytest.raises(
        ValueError, match="If `by` is specified, `by_left` and `by_right` should be None."
    ):
        df.join_asof(df, on="antananarivo", by_left="bob", by_right="bob", by="bob")

    with pytest.raises(
        ValueError,
        match="Can not specify only `by_left` or `by_right`, you need to specify both.",
    ):
        df.join_asof(df, on="antananarivo", by_left="bob")

    with pytest.raises(
        ValueError,
        match="Can not specify only `by_left` or `by_right`, you need to specify both.",
    ):
        df.join_asof(df, on="antananarivo", by_right="bob")

    with pytest.raises(
        ValueError, match="If `by` is specified, `by_left` and `by_right` should be None."
    ):
        df.join_asof(df, on="antananarivo", by_left="bob", by="bob")

    with pytest.raises(
        ValueError, match="If `by` is specified, `by_left` and `by_right` should be None."
    ):
        df.join_asof(df, on="antananarivo", by_right="bob", by="bob")

    with pytest.raises(
        ValueError, match="`by_left` and `by_right` must have the same length."
    ):
        df.join_asof(
            df,
            on="antananarivo",
            by_left=["antananarivo", "bob"],
            by_right=["antananarivo"],
        )


def test_join_duplicate_column_names(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    exception: type[Exception]

    if "polars" in str(constructor) and POLARS_VERSION < (1, 26):
        pytest.skip()
    if (
        "cudf" in str(constructor)
        # TODO(unassigned): cudf doesn't raise here for some reason,
        # need to investigate.
    ):
        request.applymarker(pytest.mark.xfail)
    if any(
        x in str(constructor)
        for x in ("pandas", "pandas[pyarrow]", "pandas[nullable]", "dask")
    ) and PANDAS_VERSION >= (3,):  # pragma: no cover
        from pandas.errors import MergeError

        exception = MergeError
    elif "sqlframe" in str(constructor):
        import duckdb

        exception = duckdb.BinderException
    elif "pyspark" in str(constructor):
        from pyspark.errors import AnalysisException

        exception = AnalysisException
    elif "modin" in str(constructor):
        exception = NotImplementedError
    elif "ibis" in str(constructor):
        # ibis doesn't raise here
        request.applymarker(pytest.mark.xfail)
    else:
        exception = nw.exceptions.DuplicateError
    df = constructor({"a": [1, 2, 3, 4, 5], "b": [6, 6, 6, 6, 6]})
    dfn = nw.from_native(df)
    with pytest.raises(exception):
        dfn.join(dfn, on=["a"]).join(dfn, on=["a"]).lazy().collect()
