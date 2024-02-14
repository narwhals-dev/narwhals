from typing import Any

def convert(df: Any, api_version: str) -> Any:
    if hasattr(df, '__polars_api_compat__'):
        return df.__polars_api_compat__()
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pass
    else:
        from polars_api_compat.pandas import convert_to_standard_compliant_dataframe as convert
        return convert(df)
