from typing import Any

def convert(df: Any, api_version: str) -> Any:
    if hasattr(df, '__polars_api_compat__'):
        return df.__polars_api_compat__()
    try:
        import polars as pl
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, pl.DataFrame):
            # can't be this simple...because, you need the `.dataframe`
            # thing to work at the end
            return df, pl
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, pd.DataFrame):
            from polars_api_compat.pandas import convert_to_standard_compliant_dataframe as convert
            return convert(df)
