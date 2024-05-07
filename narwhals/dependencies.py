# pandas / Polars / etc. : if a user passes a dataframe from one of these
# libraries, it means they must already have imported the given module.
# So, we can just check sys.modules.

import sys
from typing import Any


def get_polars() -> Any:
    """Import Polars (if available - else return None)."""
    return sys.modules.get("polars", None)


def get_pandas() -> Any:
    """Import pandas (if available - else return None)."""
    return sys.modules.get("pandas", None)


def get_modin() -> Any:  # pragma: no cover
    modin = sys.modules.get("modin", None)
    if modin is not None:
        return modin.pandas
    return None


def get_cudf() -> Any:
    return sys.modules.get("cudf", None)


def get_pyarrow() -> Any:
    return sys.modules.get("pyarrow", None)
