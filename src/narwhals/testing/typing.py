from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    import pandas as pd

    from narwhals import DataFrame, LazyFrame
    from narwhals.typing import IntoDataFrame, IntoLazyFrame


__all__ = (
    "ConstructorProtocol",
    "Data",
    "DataFrameConstructor",
    "FrameConstructor",
    "LazyFrameConstructor",
    "NarwhalsNamespace",
    "PandasConstructor",
)

Data: TypeAlias = dict[str, Any]  # TODO(Unassined): This should have a better annotation
"""A column-oriented mapping used as input to a frame constructor."""


class NarwhalsNamespace(Protocol):
    """Minimal specs of a narwhals namespace (e.g. `narwhals`, `narwhals.stable.v1`)."""

    from_native: Callable[..., Any]


FrameT_co = TypeVar("FrameT_co", bound="DataFrame[Any] | LazyFrame[Any]", covariant=True)
"""A narwhals frame type produced by a constructor (covariant)."""


class ConstructorProtocol(Protocol[FrameT_co]):
    """Interface of a frame constructor, generic over the narwhals frame it returns.

    Implemented by [`narwhals.testing.frame_constructor`][], which inherits the
    member docstrings defined here. Use the parametrized aliases
    ([`FrameConstructor`][narwhals.testing.typing.FrameConstructor],
    [`DataFrameConstructor`][narwhals.testing.typing.DataFrameConstructor],
    [`LazyFrameConstructor`][narwhals.testing.typing.LazyFrameConstructor],
    [`PandasConstructor`][narwhals.testing.typing.PandasConstructor]) to annotate
    test parameters.
    """

    is_eager: bool
    """Whether the backend returns an eager dataframe."""

    nan_is_null: bool
    """Whether floating-point NaN values are considered null."""

    needs_gpu: bool
    """Whether the backend requires GPU hardware."""

    default_include: bool
    """Whether this backend is included by default when running `--all-nw-backends`."""

    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

    def __call__(
        self, obj: Data, /, namespace: NarwhalsNamespace | None = None, **kwds: Any
    ) -> FrameT_co:
        """Build a native frame and wrap it with `namespace.from_native`.

        Arguments:
            obj: Column-oriented mapping passed to the wrapped builder.
            namespace: A narwhals namespace (e.g. `narwhals`, `narwhals.stable.v1`)
                whose `from_native` performs the wrapping.
                Defaults to the main `narwhals` namespace.
            **kwds: Forwarded to the wrapped builder.
        """
        ...

    @property
    def identifier(self) -> str:
        """Instance-level string identifier for test IDs."""
        ...

    @property
    def is_lazy(self) -> bool:
        """Whether this constructor produces a lazy native frame."""
        ...

    @property
    def is_pandas(self) -> bool:
        """Whether this is one of the pandas constructors."""
        ...

    @property
    def is_modin(self) -> bool:
        """Whether this is one of the modin constructors."""
        ...

    @property
    def is_cudf(self) -> bool:
        """Whether this is the cudf constructor."""
        ...

    @property
    def is_pandas_like(self) -> bool:
        """Whether this constructor produces a pandas-like dataframe (pandas, modin, cudf)."""
        ...

    @property
    def is_polars(self) -> bool:
        """Whether this is one of the polars constructors."""
        ...

    @property
    def is_pyarrow(self) -> bool:
        """Whether this is the pyarrow table constructor."""
        ...

    @property
    def is_dask(self) -> bool:
        """Whether this is the dask constructor."""
        ...

    @property
    def is_duckdb(self) -> bool:
        """Whether this is the duckdb constructor."""
        ...

    @property
    def is_pyspark(self) -> bool:
        """Whether this is one of the pyspark constructors."""
        ...

    @property
    def is_sqlframe(self) -> bool:
        """Whether this is the sqlframe constructor."""
        ...

    @property
    def is_ibis(self) -> bool:
        """Whether this is the ibis constructor."""
        ...

    @property
    def is_spark_like(self) -> bool:
        """Whether this constructor uses a spark-like backend (pyspark, sqlframe)."""
        ...

    @property
    def needs_pyarrow(self) -> bool:
        """Whether this constructor requires `pyarrow` to be installed."""
        ...

    @property
    def is_available(self) -> bool:
        """Whether every package this constructor needs is importable."""
        ...


FrameConstructor: TypeAlias = (
    "ConstructorProtocol[DataFrame[IntoDataFrame] | LazyFrame[IntoLazyFrame]]"
)
"""Type alias for a constructor that returns a native eager or lazy frame."""

DataFrameConstructor: TypeAlias = "ConstructorProtocol[DataFrame[IntoDataFrame]]"
"""Type alias for a constructor that returns an eager native dataframe."""

LazyFrameConstructor: TypeAlias = "ConstructorProtocol[LazyFrame[IntoLazyFrame]]"
"""Type alias for a constructor that returns a lazy native frame."""

PandasConstructor: TypeAlias = "ConstructorProtocol[DataFrame[pd.DataFrame]]"
"""Type alias for a constructor whose native frame is treated as a `pandas.DataFrame`.

Use it in pandas-specific tests so `.to_native()` exposes the concrete `pandas`
API (`.columns.name`, `.dtypes`, ...). At runtime such a test may also receive a
pandas-*like* frame (modin, cudf); the annotation is a deliberate simplification.
"""
