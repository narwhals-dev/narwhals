# ruff: noqa
import sys

import narwhals as nw
from narwhals._arrow.dataframe import ArrowDataFrame

MISSING = [
    "DataFrame.collect",
    "DataFrame.drop",
    "DataFrame.drop_nulls",
    "DataFrame.filter",
    "DataFrame.group_by",
    "DataFrame.head",
    "DataFrame.is_duplicated",
    "DataFrame.is_empty",
    "DataFrame.is_unique",
    "DataFrame.item",
    "DataFrame.iter_rows",
    "DataFrame.join",
    "DataFrame.lazy",
    "DataFrame.null_count",
    "DataFrame.pipe",
    "DataFrame.rename",
    "DataFrame.sort",
    "DataFrame.tail",
    "DataFrame.to_dict",
    "DataFrame.to_numpy",
    "DataFrame.to_pandas",
    "DataFrame.unique",
    "DataFrame.with_columns",
    "DataFrame.with_row_index",
    "DataFrame.write_parquet",
    "Series.all",
    "Series.any",
    "Series.cast",
    "Series.cat",
    "Series.diff",
    "Series.drop_nulls",
    "Series.fill_null",
    "Series.filter",
    "Series.from_iterable",
    "Series.head",
    "Series.is_between",
    "Series.is_duplicated",
    "Series.is_empty",
    "Series.is_first_distinct",
    "Series.is_in",
    "Series.is_last_distinct",
    "Series.is_null",
    "Series.is_sorted",
    "Series.is_unique",
    "Series.item",
    "Series.len",
    "Series.max",
    "Series.mean",
    "Series.min",
    "Series.n_unique",
    "Series.null_count",
    "Series.quantile",
    "Series.round",
    "Series.sample",
    "Series.shift",
    "Series.sort",
    "Series.std",
    "Series.str",
    "Series.sum",
    "Series.tail",
    "Series.to_frame",
    "Series.to_pandas",
    "Series.unique",
    "Series.value_counts",
    "Series.zip_with",
]


class MockDataFrame:
    # Make a little mock object so we can instantiate
    # PandasDataFrame without having pandas installed
    def __init__(self, dataframe): ...

    def __narwhals_dataframe__(self):
        return self

    @property
    def columns(self):
        return []

    @property
    def loc(self):
        return self

    def __getitem__(self, *args):
        return MockSeries(self)


class MockSeries:
    # Make a little mock object so we can instantiate
    # nw.DataFrame without having dataframe libraries
    # installed
    def __init__(self, series): ...

    def __narwhals_series__(self):
        return self

    @property
    def name(self):
        return "a"


if __name__ == "__main__":
    missing = []

    df_pa = ArrowDataFrame(MockDataFrame({"a": [1, 2, 3]}))
    df_pd = nw.DataFrame(MockDataFrame({"a": [1, 2, 3]}), is_polars=True)
    pa_methods = [f"DataFrame.{x}" for x in df_pa.__dir__() if not x.startswith("_")]
    pd_methods = [f"DataFrame.{x}" for x in df_pd.__dir__() if not x.startswith("_")]
    missing.extend([x for x in pd_methods if x not in pa_methods and x not in MISSING])

    ser_pa = df_pa["a"]
    ser_pd = df_pd["a"]
    pa_methods = [f"Series.{x}" for x in ser_pa.__dir__() if not x.startswith("_")]
    pd_methods = [f"Series.{x}" for x in ser_pd.__dir__() if not x.startswith("_")]
    missing.extend([x for x in pd_methods if x not in pa_methods and x not in MISSING])

    if missing:
        print(sorted(missing))
        sys.exit(1)
    sys.exit(0)
