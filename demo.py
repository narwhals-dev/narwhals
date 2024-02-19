# ruff: noqa
import polars as pl
from great_tables.data import sp500

from puffin import to_original_object
from puffin import to_polars_api

# Define the start and end dates for the data range
start_date = "2010-06-07"
end_date = "2010-06-14"

# Filter sp500 using Pandas to dates between `start_date` and `end_date`
# sp500_mini = sp500[(sp500["date"] >= start_date) & (sp500["date"] <= end_date)]


def dataframe_agnostic_filter(df_raw, start_date, end_date):
    # opt-in to Polars API
    df, pl = to_polars_api(df_raw, version="0.20")

    # Use (supported subset of) Polars API
    df = df.filter(
        pl.col("date") >= start_date,
        pl.col("date") <= end_date,
    )

    # Return underlying dataframe (same class passed by user)
    return to_original_object(df)


sp500_mini = dataframe_agnostic_filter(sp500, start_date, end_date)
print(pl.from_pandas(sp500_mini))
