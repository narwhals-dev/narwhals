from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.functions.col import col

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr

__all__ = ("max", "mean", "median", "min", "sum")


def max(*columns: str) -> Expr:
    return col(columns).max()


def mean(*columns: str) -> Expr:
    return col(columns).mean()


def min(*columns: str) -> Expr:
    return col(columns).min()


def median(*columns: str) -> Expr:
    return col(columns).median()


def sum(*columns: str) -> Expr:
    return col(columns).sum()
