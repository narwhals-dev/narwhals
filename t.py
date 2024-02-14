import pandas as pd
import polars_api_compat
import polars as pl
from datetime import datetime

df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
dfx, plx = polars_api_compat.convert(df, api_version='0.20')
result = dfx.with_columns(
    c = plx.col('a') + plx.col('b'),
    d = plx.col('a') - plx.col('a').mean(),
)
print(result.dataframe)
result = dfx.with_columns(
    plx.all() * 2,
)
print(result.dataframe)
