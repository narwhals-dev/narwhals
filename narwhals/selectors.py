from __future__ import annotations

from typing import Any

from narwhals.expr import Expr
from narwhals.utils import flatten


class Selector(Expr): ...


def by_dtype(*dtypes: Any) -> Expr:
    """Select columns based on their dtype.

    Arguments:
        dtypes: one or data types to select

    Returns:
        A new expression.
    """
    return Selector(lambda plx: plx.selectors.by_dtype(flatten(dtypes)))


def numeric() -> Expr:
    """Select numeric columns.

    Returns:
        A new expression.
    """
    return Selector(lambda plx: plx.selectors.numeric())


def boolean() -> Expr:
    """Select boolean columns.

    Returns:
        A new expression.
    """
    return Selector(lambda plx: plx.selectors.boolean())


def string() -> Expr:
    """Select string columns.

    Returns:
        A new expression.
    """
    return Selector(lambda plx: plx.selectors.string())


def categorical() -> Expr:
    """Select categorical columns.

    Returns:
        A new expression.
    """
    return Selector(lambda plx: plx.selectors.categorical())


def all() -> Expr:
    """Select all columns.

    Returns:
        A new expression.
    """
    return Selector(lambda plx: plx.selectors.all())


__all__ = [
    "all",
    "boolean",
    "by_dtype",
    "categorical",
    "numeric",
    "string",
]
