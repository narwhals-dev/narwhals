from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    import pandas as pd

    from narwhals.testing.constructors import frame_constructor
    from narwhals.typing import IntoDataFrame, IntoFrame, IntoLazyFrame


__all__ = (
    "Data",
    "DataFrameConstructor",
    "FrameConstructor",
    "LazyFrameConstructor",
    "PandasConstructor",
)

FrameConstructor: TypeAlias = "frame_constructor[IntoFrame]"
"""Type alias for a constructor that returns a native eager or lazy frame."""

DataFrameConstructor: TypeAlias = "frame_constructor[IntoDataFrame]"
"""Type alias for a constructor that returns an eager native dataframe."""

LazyFrameConstructor: TypeAlias = "frame_constructor[IntoLazyFrame]"
"""Type alias for a constructor that returns a lazy native frame."""

PandasConstructor: TypeAlias = "frame_constructor[pd.DataFrame]"
"""Type alias for a constructor whose native frame is treated as a `pandas.DataFrame`.

Use it in pandas-specific tests so `.to_native()` exposes the concrete `pandas`
API (`.columns.name`, `.dtypes`, ...). At runtime such a test may also receive a
pandas-*like* frame (modin, cudf); the annotation is a deliberate simplification.
"""

Data: TypeAlias = dict[str, Any]  # TODO(Unassined): This should have a better annotation
"""A column-oriented mapping used as input to a frame constructor."""


class NarwhalsNamespace(Protocol):
    """Minimal specs of a narwhals namespace (e.g. `narwhals`, `narwhals.stable.v1`)."""

    from_native: Callable[..., Any]
