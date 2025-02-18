from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from typing import Literal

import pandas as pd
import pytest

import narwhals as nw_main  # use nw_main in some tests for coverage
import narwhals.stable.v1 as nw
from narwhals.utils import Implementation
from tests.utils import DUCKDB_VERSION
from tests.utils import PANDAS_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_inner_join_two_keys(constructor: Constructor) -> None:
    data = {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zor ro": [7.0, 8.0, 9.0],
        "idx": [0, 1, 2],
    }
    df = nw_main.from_native(constructor(data))
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
        df_right,  # type: ignore[arg-type]
        left_on="antananarivo",
        right_on="antananarivo",
        how="inner",
    ).sort("idx")
    result_on = df.join(df_right, on="antananarivo", how="inner").sort("idx")  # type: ignore[arg-type]
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


def test_cross_join(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 1, 4):
        request.applymarker(pytest.mark.xfail)
    data = {"antananarivo": [1, 3, 2]}
    df = nw.from_native(constructor(data))
    result = df.join(df, how="cross").sort("antananarivo", "antananarivo_right")  # type: ignore[arg-type]
    expected = {
        "antananarivo": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "antananarivo_right": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    }
    assert_equal_data(result, expected)

    with pytest.raises(
        ValueError, match="Can not pass `left_on`, `right_on` or `on` keys for cross join"
    ):
        df.join(df, how="cross", left_on="antananarivo")  # type: ignore[arg-type]


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("suffix", ["_right", "_custom_suffix"])
def test_suffix(constructor: Constructor, how: str, suffix: str) -> None:
    data = {
        "antananarivo": [1, 3, 2],
        "bob": [4, 4, 6],
        "zor ro": [7.0, 8.0, 9.0],
    }
    df = nw.from_native(constructor(data))
    df_right = df
    result = df.join(
        df_right,  # type: ignore[arg-type]
        left_on=["antananarivo", "bob"],
        right_on=["antananarivo", "bob"],
        how=how,  # type: ignore[arg-type]
        suffix=suffix,
    )
    result_cols = result.collect_schema().names()
    assert result_cols == ["antananarivo", "bob", "zor ro", f"zor ro{suffix}"]


@pytest.mark.parametrize("suffix", ["_right", "_custom_suffix"])
def test_cross_join_suffix(
    constructor: Constructor, suffix: str, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 1, 4):
        request.applymarker(pytest.mark.xfail)
    data = {"antananarivo": [1, 3, 2]}
    df = nw.from_native(constructor(data))
    result = df.join(df, how="cross", suffix=suffix).sort(  # type: ignore[arg-type]
        "antananarivo", f"antananarivo{suffix}"
    )
    expected = {
        "antananarivo": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        f"antananarivo{suffix}": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    }
    assert_equal_data(result, expected)


def test_cross_join_non_pandas() -> None:
    data = {"antananarivo": [1, 3, 2]}
    df = nw.from_native(pd.DataFrame(data))
    # HACK to force testing for a non-pandas codepath
    df._compliant_frame._implementation = Implementation.MODIN
    result = df.join(df, how="cross")  # type: ignore[arg-type]
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
    result = df.join(other, how="anti", left_on=join_key, right_on=join_key)  # type: ignore[arg-type]
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
    result = df.join(other, how="semi", left_on=join_key, right_on=join_key).sort(  # type: ignore[arg-type]
        "antananarivo"
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize("how", ["right", "full"])
def test_join_not_implemented(constructor: Constructor, how: str) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            f"Only the following join strategies are supported: ('inner', 'left', 'cross', 'anti', 'semi'); found '{how}'."
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
    result = df_left.join(df_right, left_on="bob", right_on="co", how="left")  # type: ignore[arg-type]
    result = result.sort("idx")
    result = result.drop("idx_right")
    expected = {
        "antananarivo": [1, 2, 3],
        "bob": [4, 5, 6],
        "idx": [0, 1, 2],
        "antananarivo_right": [1, 2, None],
    }
    result_on_list = df_left.join(
        df_right,  # type: ignore[arg-type]
        on=["antananarivo", "idx"],
        how="left",
    )
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
        df_right,  # type: ignore[arg-type]
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
    result = df_left.join(df_right, left_on="bob", right_on="c", how="left").sort("idx")  # type: ignore[arg-type]
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
    result = df_left.join(
        df_right,  # type: ignore[arg-type]
        left_on="antananarivo",
        right_on="d",
        how="left",
    )
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


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (
            "backward",
            {
                "antananarivo": [1, 5, 10],
                "val": ["a", "b", "c"],
                "val_right": [1, 3, 7],
            },
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
            {
                "antananarivo": [1, 5, 10],
                "val": ["a", "b", "c"],
                "val_right": [1, 6, 7],
            },
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
    if "duckdb" in str(constructor) and strategy == "nearest":
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
        df_right,  # type: ignore[arg-type]
        left_on="antananarivo",
        right_on="antananarivo",
        strategy=strategy,
    )
    result_on = df.join_asof(df_right, on="antananarivo", strategy=strategy)  # type: ignore[arg-type]
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
    if "duckdb" in str(constructor) and strategy == "nearest":
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
        df_right,  # type: ignore[arg-type]
        left_on="datetime",
        right_on="datetime",
        strategy=strategy,
    )
    result_on = df.join_asof(df_right, on="datetime", strategy=strategy)  # type: ignore[arg-type]
    assert_equal_data(result.sort(by="datetime"), expected)
    assert_equal_data(result_on.sort(by="datetime"), expected)


def test_joinasof_by(
    constructor: Constructor,
    request: pytest.FixtureRequest,
) -> None:
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
    result = df.join_asof(df_right, on="antananarivo", by_left="bob", by_right="bob")  # type: ignore[arg-type]
    result_by = df.join_asof(df_right, on="antananarivo", by="bob")  # type: ignore[arg-type]
    expected = {
        "antananarivo": [1, 5, 7, 10],
        "bob": ["D", "D", "C", "A"],
        "c": [9, 2, 1, 1],
        "d": [1, 3, None, 4],
    }
    assert_equal_data(result.sort(by="antananarivo"), expected)
    assert_equal_data(result_by.sort(by="antananarivo"), expected)


def test_joinasof_suffix(
    constructor: Constructor,
    request: pytest.FixtureRequest,
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
        df_right,  # type: ignore[arg-type]
        left_on="antananarivo",
        right_on="antananarivo",
        suffix="_y",
    )
    expected = {
        "antananarivo": [1, 5, 10],
        "val": ["a", "b", "c"],
        "val_y": [1, 3, 7],
    }
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
            df,  # type: ignore[arg-type]
            left_on="antananarivo",
            right_on="antananarivo",
            strategy=strategy,
        )


def test_joinasof_keys_exceptions(constructor: Constructor) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))

    with pytest.raises(
        ValueError,
        match=r"Either \(`left_on` and `right_on`\) or `on` keys should be specified.",
    ):
        df.join_asof(df, left_on="antananarivo")  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=r"Either \(`left_on` and `right_on`\) or `on` keys should be specified.",
    ):
        df.join_asof(df, right_on="antananarivo")  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=r"Either \(`left_on` and `right_on`\) or `on` keys should be specified.",
    ):
        df.join_asof(df)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match="If `on` is specified, `left_on` and `right_on` should be None.",
    ):
        df.join_asof(
            df,  # type: ignore[arg-type]
            left_on="antananarivo",
            right_on="antananarivo",
            on="antananarivo",
        )
    with pytest.raises(
        ValueError,
        match="If `on` is specified, `left_on` and `right_on` should be None.",
    ):
        df.join_asof(df, left_on="antananarivo", on="antananarivo")  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match="If `on` is specified, `left_on` and `right_on` should be None.",
    ):
        df.join_asof(df, right_on="antananarivo", on="antananarivo")  # type: ignore[arg-type]


def test_joinasof_by_exceptions(constructor: Constructor) -> None:
    data = {"antananarivo": [1, 3, 2], "bob": [4, 4, 6], "zor ro": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    with pytest.raises(
        ValueError,
        match="If `by` is specified, `by_left` and `by_right` should be None.",
    ):
        df.join_asof(df, on="antananarivo", by_left="bob", by_right="bob", by="bob")  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="Can not specify only `by_left` or `by_right`, you need to specify both.",
    ):
        df.join_asof(df, on="antananarivo", by_left="bob")  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="Can not specify only `by_left` or `by_right`, you need to specify both.",
    ):
        df.join_asof(df, on="antananarivo", by_right="bob")  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="If `by` is specified, `by_left` and `by_right` should be None.",
    ):
        df.join_asof(df, on="antananarivo", by_left="bob", by="bob")  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="If `by` is specified, `by_left` and `by_right` should be None.",
    ):
        df.join_asof(df, on="antananarivo", by_right="bob", by="bob")  # type: ignore[arg-type]


def test_join_duplicate_column_names(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if (
        "polars" in str(constructor)  # https://github.com/pola-rs/polars/issues/21048
        or "cudf" in str(constructor)
        # TODO(unassigned): cudf doesn't raise here for some reason,
        # need to investigate.
    ):
        request.applymarker(pytest.mark.xfail)
    elif "modin" in str(constructor):
        exception = NotImplementedError
    else:
        exception = nw.exceptions.DuplicateError  # type: ignore[assignment]
    df = constructor({"a": [1, 2, 3, 4, 5], "b": [6, 6, 6, 6, 6]})
    dfn = nw.from_native(df)
    with pytest.raises(exception):
        dfn.join(dfn, on=["a"]).join(dfn, on=["a"])  # type: ignore[arg-type]
