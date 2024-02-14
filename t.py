import pandas as pd
import polars_api_compat
import polars as pl
from datetime import datetime

df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})

dfx, plx = polars_api_compat.convert(df, api_version='0.20')

print(dfx.with_columns((plx.col('a') + plx.col('b')).alias('c')).dataframe)
print(dfx.with_columns((plx.col('a') - plx.col('a').mean()).alias('c')).dataframe)
print(dfx.with_columns(
    (plx.col('a') - plx.col('a').mean()).alias('c'),
    (plx.col('a') + plx.col('a').mean()).alias('d'),
).dataframe)
print(dfx.with_columns([
    (plx.col('a') - plx.col('a').mean()).alias('c'),
    (plx.col('a') + plx.col('a').mean()).alias('d'),
]).dataframe)
print(dfx.select(e=plx.col('a') - plx.col('a').mean()).dataframe)
print(dfx.select(plx.all() + 1).dataframe)
