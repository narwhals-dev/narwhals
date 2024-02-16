from typing import Any


def convert(df: Any, version: str) -> Any:
    if hasattr(df, "__polars_api_compat__"):
        return df.__polars_api_compat__()
    try:
        import polars as pl
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, pl.DataFrame):
            from polars_api_compat.polars import convert

            return convert(df)
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, pd.DataFrame):
            from polars_api_compat.pandas import (
                convert_to_standard_compliant_dataframe as convert,
            )

            return convert(df)
