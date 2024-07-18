"""
Hopefully temporary script which tracks which methods we're missing
for the PyArrow table backend.

If you implement a method, please remove it from the `MISSING` list.
"""

# ruff: noqa
import sys

import narwhals as nw
from narwhals._arrow.dataframe import ArrowDataFrame

MISSING = [
    "DataFrame.is_duplicated",
    "DataFrame.is_unique",
    "DataFrame.iter_rows",
    "DataFrame.pipe",
    "DataFrame.unique",
    "Series.drop_nulls",
    "Series.fill_null",
    "Series.from_iterable",
    "Series.is_between",
    "Series.is_duplicated",
    "Series.is_first_distinct",
    "Series.is_last_distinct",
    "Series.is_sorted",
    "Series.is_unique",
    "Series.n_unique",
    "Series.quantile",
    "Series.round",
    "Series.shift",
    "Series.sort",
    "Series.to_frame",
    "Series.to_pandas",
    "Series.unique",
    "Series.value_counts",
]


class MockDataFrame:
    # Make a little mock object so we can instantiate
    # PandasLikeDataFrame without having pandas installed
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
    no_longer_missing = []

    df_pa = ArrowDataFrame(MockDataFrame({"a": [1, 2, 3]}), backend_version=(13, 0))
    df_nw = nw.DataFrame(
        MockDataFrame({"a": [1, 2, 3]}),
        is_polars=True,
        backend_version=(1,),
        level="full",
    )
    pa_methods = [f"DataFrame.{x}" for x in df_pa.__dir__() if not x.startswith("_")]
    nw_methods = [f"DataFrame.{x}" for x in df_nw.__dir__() if not x.startswith("_")]
    missing.extend(
        [
            x
            for x in nw_methods
            if x not in pa_methods and x not in MISSING and x not in {"level"}
        ]
    )
    no_longer_missing.extend([x for x in MISSING if x in pa_methods and x in nw_methods])

    ser_pa = df_pa["a"]
    ser_pd = df_nw["a"]
    pa_methods = [f"Series.{x}" for x in ser_pa.__dir__() if not x.startswith("_")]
    nw_methods = [f"Series.{x}" for x in ser_pd.__dir__() if not x.startswith("_")]
    missing.extend([x for x in nw_methods if x not in pa_methods and x not in MISSING])
    no_longer_missing.extend([x for x in MISSING if x in pa_methods and x in nw_methods])

    if missing:
        print(
            "The following have not been implemented for the Arrow backend: ",
            sorted(missing),
        )
        sys.exit(1)

    if no_longer_missing:
        print(
            "Please remove the following from MISSING in utils/check_backend_completeness.py: ",
            sorted(no_longer_missing),
        )
        sys.exit(1)

    sys.exit(0)
