from __future__ import annotations

import sys
from functools import partial
from functools import wraps
from typing import Any
from typing import Callable
from typing import TypeVar

from narwhals.translate import from_native
from narwhals.translate import to_native

if sys.version_info >= (3, 10):
    from typing import Concatenate  # pragma: no cover
    from typing import ParamSpec  # pragma: no cover
else:
    from typing_extensions import Concatenate  # pragma: no cover
    from typing_extensions import ParamSpec  # pragma: no cover

T = TypeVar("T")
PS = ParamSpec("PS")


def narwhalify(
    func: Callable[Concatenate[T, PS], T] | None = None,
    from_kwargs: dict[str, Any] | None = None,
    to_kwargs: dict[str, Any] | None = None,
) -> Callable[Concatenate[Any, PS], Any] | Callable[[Any], Any]:
    """Decorator that wraps a dataframe agnostic function between `from_native` and `to_native` conversion.

    All the expressions used within `func` must be `narwhals` compatible, and we assume that only one
    dataframe/series-like is returned from it.

    Warning:
        `func` can take an arbitrary number of dataframe/series objects which will be converted using `from_native`.
        However, as these arguments are parsed as positional, make sure to:

        - Pass them as the first positional arguments.
        - Explicitly pass any other argument as keyword.

        Check examples section below to see an instance of that.

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

        >>> @nw.narwhalify
        ... def join_on_key(left, right, key):
        ...     return left.join(right, left_on=key, right_on=key)

        >>> frame1 = pl.DataFrame({'a': [1, 1, 2], 'b': [0, 1, 2]})
        >>> frame2 = pl.DataFrame({'a': [1, 2], 'c': ['x', 'y']})

        >>> join_on_key(frame1, frame2, key='a')
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 0   ┆ x   │
        │ 1   ┆ 1   ┆ x   │
        │ 2   ┆ 2   ┆ y   │
        └─────┴─────┴─────┘

        >>> join_on_key(frame1, frame2, 'a')  # tries to convert 'a' as well using nw.from_native
        Traceback (most recent call last):
        ...
        TypeError: Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe, got: <class 'str'>

        >>> join_on_key(frame1, right=frame2, key='a')  # does not convert frame2 because passed as keyword
        Traceback (most recent call last):
        ...
        AttributeError: 'DataFrame' object has no attribute '_is_polars'
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
    def wrapper(*frames: Any, **kwargs: PS.kwargs) -> Any:
        nw_frames = [from_native(frame, **from_kwargs) for frame in frames]
        result = func(*nw_frames, **kwargs)
        return to_native(result, **to_kwargs)

    return wrapper
