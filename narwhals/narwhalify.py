from __future__ import annotations

from functools import partial
from functools import wraps
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import TypeVar

from narwhals.translate import from_native
from narwhals.translate import to_native

if TYPE_CHECKING:
    from typing_extensions import Concatenate  # pragma: no cover
    from typing_extensions import ParamSpec  # pragma: no cover

    T = TypeVar("T")
    PS = ParamSpec("PS")


def narwhalify(
    func: Callable[Concatenate[T, PS], Any] | None = None,
    from_kwargs: dict[str, Any] | None = None,
    to_kwargs: dict[str, Any] | None = None,
) -> Callable[Concatenate[Any, PS], Any] | Callable[[Any], Any]:
    """Decorator that wraps a dataframe agnostic function between `from_native` and `to_native` conversion.

    All the expressions used within `func` must be `narwhals` compatible.

    Warning:
        We are making a few assumptions on `func`:

        - The first position argument is a passed through `from_native`
        - Then `func` is called as `func(nw_frame, *args, **kwargs)`
        - Finally the result of `func` is passed to `to_native` and returned.

    Arguments:
        func: Narwhals compatible function to be wrapped
        from_kwargs: keyword arguments to pass to `from_native` transformation
        to_kwargs: keyword arguments to pass to `to_native` transformation

    Returns:
        Decorated function.

    Examples:
        >>> import narwhals as nw

        >>> @nw.narwhalify
        ... def func(df_any):
        ...     return df_any.select(
        ...         a_sum=nw.col('a').sum(),
        ...         a_mean=nw.col('a').mean(),
        ...         a_std=nw.col('a').std(),
        ...     )

        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 1, 2]})
        >>> func(df)
           a_sum    a_mean    a_std
        0      4  1.333333  0.57735

        >>> import polars as pl
        >>> df = pl.DataFrame({'a': [1, 1, 2]})
        >>> func(df)
        shape: (1, 3)
        ┌───────┬──────────┬─────────┐
        │ a_sum ┆ a_mean   ┆ a_std   │
        │ ---   ┆ ---      ┆ ---     │
        │ i64   ┆ f64      ┆ f64     │
        ╞═══════╪══════════╪═════════╡
        │ 4     ┆ 1.333333 ┆ 0.57735 │
        └───────┴──────────┴─────────┘

        >>> @nw.narwhalify(
        ...     from_kwargs={'eager_only': True},
        ...     to_kwargs={'strict': False},
        ... )
        ... def shape_greater_than(df_any, n=0):
        ...     return df_any.shape[0] > n

        >>> shape_greater_than(df)
        True
        >>> shape_greater_than(df, 5)
        False
        >>> shape_greater_than(df_any=df, n=5)
        Traceback (most recent call last):
        ...
        TypeError: shape_greater_than() missing 1 required positional argument: 'frame'
        >>> shape_greater_than(pl.LazyFrame(df))
        Traceback (most recent call last):
        ...
        TypeError: Cannot only use `eager_only` with polars.LazyFrame

    """
    if func is None:
        return partial(narwhalify, from_kwargs=from_kwargs, to_kwargs=to_kwargs)

    from_kwargs = from_kwargs or {
        "strict": True,
        "eager_only": None,
        "series_only": None,
        "allow_series": None,
    }
    to_kwargs = to_kwargs or {"strict": True}

    @wraps(func)
    def wrapper(frame: Any, *args: PS.args, **kwargs: PS.kwargs) -> Any:
        nw_frame = from_native(frame, **from_kwargs)
        result = func(nw_frame, *args, **kwargs)
        return to_native(result, **to_kwargs)

    return wrapper
