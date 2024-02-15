import pandas as pd
import polars_api_compat
import polars as pl
from datetime import datetime

df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6], 'c': [4., 6.5, 1.2]})

def my_function(df):
    dfx, plx = polars_api_compat.convert(df, api_version='0.20')
    result = dfx.with_columns(
        d = (plx.col('b') + plx.col('c')) ** plx.col('a')
    )
    return result.dataframe

print(my_function(df))
