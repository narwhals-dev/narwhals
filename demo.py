from great_tables.data import sp500
from polars_api_compat import translate

# Define the start and end dates for the data range
start_date = "2010-06-07"
end_date = "2010-06-14"

# Filter sp500 using Pandas to dates between `start_date` and `end_date`
# sp500_mini = sp500[(sp500["date"] >= start_date) & (sp500["date"] <= end_date)]


def dataframe_agnostic_filter(df, start_date, end_date):
    # opt-in to Polars API
    dfx, plx = translate(df, version="0.20")

    # Use (supported subset of) Polars API
    dfx = dfx.filter(
        plx.col("date") >= start_date,
        plx.col("date") <= end_date,
    )

    # Return underlying dataframe (same class passed by user)
    return dfx.dataframe


sp500_mini = dataframe_agnostic_filter(sp500, start_date, end_date)
print(sp500_mini)
