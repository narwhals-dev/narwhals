from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import NoReturn

from narwhals.expr import Expr
from narwhals.utils import flatten

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType


class Selector(Expr):
    def _to_expr(self: Self) -> Expr:
        return Expr(
            to_compliant_expr=self._to_compliant_expr,
            is_order_dependent=self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def __add__(self: Self, other: Any) -> Expr:  # type: ignore[override]
        if isinstance(other, Selector):
            msg = "unsupported operand type(s) for op: ('Selector' + 'Selector')"
            raise TypeError(msg)
        return self._to_expr() + other  # type: ignore[no-any-return]

    def __rsub__(self: Self, other: Any) -> NoReturn:
        raise NotImplementedError

    def __rand__(self: Self, other: Any) -> NoReturn:
        raise NotImplementedError

    def __ror__(self: Self, other: Any) -> NoReturn:
        raise NotImplementedError


def by_dtype(*dtypes: DType | type[DType] | Iterable[DType | type[DType]]) -> Selector:
    """Select columns based on their dtype.

    Arguments:
        dtypes: one or data types to select

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [4.1, 2.3]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function to select int64 and float64
        dtypes and multiplies each value by 2:

        >>> def agnostic_select_by_dtype(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = nw.from_native(df_native)
        ...     return df_nw.select(ncs.by_dtype(nw.Int64, nw.Float64) * 2).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_select_by_dtype`:

        >>> agnostic_select_by_dtype(df_pd)
           a    c
        0  2  8.2
        1  4  4.6

        >>> agnostic_select_by_dtype(df_pl)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ c   │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 2   ┆ 8.2 │
        │ 4   ┆ 4.6 │
        └─────┴─────┘

        >>> agnostic_select_by_dtype(df_pa)
        pyarrow.Table
        a: int64
        c: double
        ----
        a: [[2,4]]
        c: [[8.2,4.6]]
    """
    return Selector(
        lambda plx: plx.selectors.by_dtype(flatten(dtypes)),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


def matches(pattern: str) -> Selector:
    """Select all columns that match the given regex pattern.

    Arguments:
        pattern: A valid regular expression pattern.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {
        ...     "foo": ["x", "y"],
        ...     "bar": [123, 456],
        ...     "baz": [2.0, 5.5],
        ...     "zap": [0, 1],
        ... }
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function to select column names
        containing an 'a', preceded by a character that is not 'z':

        >>> def agnostic_select_match(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = nw.from_native(df_native)
        ...     return df_nw.select(ncs.matches("[^z]a")).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_select_match`:

        >>> agnostic_select_match(df_pd)
           bar  baz
        0  123  2.0
        1  456  5.5

        >>> agnostic_select_match(df_pl)
        shape: (2, 2)
        ┌─────┬─────┐
        │ bar ┆ baz │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 123 ┆ 2.0 │
        │ 456 ┆ 5.5 │
        └─────┴─────┘

        >>> agnostic_select_match(df_pa)
        pyarrow.Table
        bar: int64
        baz: double
        ----
        bar: [[123,456]]
        baz: [[2,5.5]]
    """
    return Selector(
        lambda plx: plx.selectors.matches(pattern),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


def numeric() -> Selector:
    """Select numeric columns.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [4.1, 2.3]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function to select numeric
        dtypes and multiplies each value by 2:

        >>> def agnostic_select_numeric(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = nw.from_native(df_native)
        ...     return df_nw.select(ncs.numeric() * 2).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_select_numeric`:

        >>> agnostic_select_numeric(df_pd)
           a    c
        0  2  8.2
        1  4  4.6

        >>> agnostic_select_numeric(df_pl)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ c   │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 2   ┆ 8.2 │
        │ 4   ┆ 4.6 │
        └─────┴─────┘

        >>> agnostic_select_numeric(df_pa)
        pyarrow.Table
        a: int64
        c: double
        ----
        a: [[2,4]]
        c: [[8.2,4.6]]
    """
    return Selector(
        lambda plx: plx.selectors.numeric(),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


def boolean() -> Selector:
    """Select boolean columns.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [False, True]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function to select boolean dtypes:

        >>> def agnostic_select_boolean(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = nw.from_native(df_native)
        ...     return df_nw.select(ncs.boolean()).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_select_boolean`:

        >>> agnostic_select_boolean(df_pd)
               c
        0  False
        1   True

        >>> agnostic_select_boolean(df_pl)
        shape: (2, 1)
        ┌───────┐
        │ c     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ false │
        │ true  │
        └───────┘

        >>> agnostic_select_boolean(df_pa)
        pyarrow.Table
        c: bool
        ----
        c: [[false,true]]
    """
    return Selector(
        lambda plx: plx.selectors.boolean(),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


def string() -> Selector:
    """Select string columns.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [False, True]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function to select string dtypes:

        >>> def agnostic_select_string(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = nw.from_native(df_native)
        ...     return df_nw.select(ncs.string()).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_select_string`:

        >>> agnostic_select_string(df_pd)
           b
        0  x
        1  y

        >>> agnostic_select_string(df_pl)
        shape: (2, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ str │
        ╞═════╡
        │ x   │
        │ y   │
        └─────┘

        >>> agnostic_select_string(df_pa)
        pyarrow.Table
        b: string
        ----
        b: [["x","y"]]
    """
    return Selector(
        lambda plx: plx.selectors.string(),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


def categorical() -> Selector:
    """Select categorical columns.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [False, True]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function that first converts column "b" to
        categorical, and then selects categorical dtypes:

        >>> def agnostic_select_categorical(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = nw.from_native(df_native).with_columns(
        ...         b=nw.col("b").cast(nw.Categorical())
        ...     )
        ...     return df_nw.select(ncs.categorical()).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_select_categorical`:

        >>> agnostic_select_categorical(df_pd)
           b
        0  x
        1  y

        >>> agnostic_select_categorical(df_pl)
        shape: (2, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ cat │
        ╞═════╡
        │ x   │
        │ y   │
        └─────┘

        >>> agnostic_select_categorical(df_pa)
        pyarrow.Table
        b: dictionary<values=string, indices=uint32, ordered=0>
        ----
        b: [  -- dictionary:
        ["x","y"]  -- indices:
        [0,1]]
    """
    return Selector(
        lambda plx: plx.selectors.categorical(),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


def all() -> Selector:
    """Select all columns.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [False, True]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function to select all dtypes:

        >>> def agnostic_select_all(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = nw.from_native(df_native)
        ...     return df_nw.select(ncs.all()).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_select_all`:

        >>> agnostic_select_all(df_pd)
           a  b      c
        0  1  x  False
        1  2  y   True

        >>> agnostic_select_all(df_pl)
        shape: (2, 3)
        ┌─────┬─────┬───────┐
        │ a   ┆ b   ┆ c     │
        │ --- ┆ --- ┆ ---   │
        │ i64 ┆ str ┆ bool  │
        ╞═════╪═════╪═══════╡
        │ 1   ┆ x   ┆ false │
        │ 2   ┆ y   ┆ true  │
        └─────┴─────┴───────┘

        >>> agnostic_select_all(df_pa)
        pyarrow.Table
        a: int64
        b: string
        c: bool
        ----
        a: [[1,2]]
        b: [["x","y"]]
        c: [[false,true]]
    """
    return Selector(
        lambda plx: plx.selectors.all(),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


__all__ = [
    "all",
    "boolean",
    "by_dtype",
    "categorical",
    "matches",
    "numeric",
    "string",
]
