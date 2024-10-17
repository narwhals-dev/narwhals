from __future__ import annotations

import math
import sys
import warnings
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Sequence

import pandas as pd

from narwhals.typing import IntoDataFrame
from narwhals.typing import IntoFrame
from narwhals.utils import Implementation

if sys.version_info >= (3, 10):
    from typing import TypeAlias  # pragma: no cover
else:
    from typing_extensions import TypeAlias  # pragma: no cover

Constructor: TypeAlias = Callable[[dict[str, list[Any]]], IntoFrame]
ConstructorEager: TypeAlias = Callable[[dict[str, list[Any]]], IntoDataFrame]


def zip_strict(left: Sequence[Any], right: Sequence[Any]) -> Iterator[Any]:
    if len(left) != len(right):
        msg = f"left {len(left)=} != right {len(right)=}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    return zip(left, right)


def compare_dicts(result: Any, expected: dict[str, list[Any]]) -> None:
    if hasattr(result, "collect"):
        result = result.collect()
    if hasattr(result, "columns"):
        for key in result.columns:
            assert key in expected
    for key in expected:
        result_key = result[key]
        if (
            hasattr(result_key, "_compliant_series")
            and result_key._compliant_series._implementation is Implementation.CUDF
        ):  # pragma: no cover
            result_key = result_key.to_pandas()
        for lhs, rhs in zip_strict(result_key, expected[key]):
            if hasattr(lhs, "as_py"):
                lhs = lhs.as_py()  # noqa: PLW2901
            if hasattr(rhs, "as_py"):  # pragma: no cover
                rhs = rhs.as_py()  # noqa: PLW2901
            if hasattr(lhs, "item"):  # pragma: no cover
                lhs = lhs.item()  # noqa: PLW2901
            if hasattr(rhs, "item"):  # pragma: no cover
                rhs = rhs.item()  # noqa: PLW2901
            if isinstance(lhs, float) and not math.isnan(lhs):
                assert math.isclose(lhs, rhs, rel_tol=0, abs_tol=1e-6), (lhs, rhs)
            elif isinstance(lhs, float) and math.isnan(lhs) and rhs is not None:
                assert math.isnan(rhs), (lhs, rhs)  # pragma: no cover
            elif pd.isna(lhs):
                assert pd.isna(rhs), (lhs, rhs)
            else:
                assert lhs == rhs, (lhs, rhs)


def maybe_get_modin_df(df_pandas: pd.DataFrame) -> Any:
    """Convert a pandas DataFrame to a Modin DataFrame if Modin is available."""
    try:
        import modin.pandas as mpd
    except ImportError:  # pragma: no cover
        return df_pandas.copy()
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return mpd.DataFrame(df_pandas.to_dict(orient="list"))


def is_windows() -> bool:
    """Check if the current platform is Windows."""
    return sys.platform in ["win32", "cygwin"]
