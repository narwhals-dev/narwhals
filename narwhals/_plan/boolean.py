from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Any
import typing as t  # noqa: F401

from narwhals._plan.common import Function


class BooleanFunction(Function): ...


class All(BooleanFunction): ...


class AllHorizontal(BooleanFunction): ...


class Any(BooleanFunction): ...


class AnyHorizontal(BooleanFunction): ...


class IsBetween(BooleanFunction): ...


class IsDuplicated(BooleanFunction): ...


class IsFinite(BooleanFunction): ...


class IsFirstDistinct(BooleanFunction): ...


class IsIn(BooleanFunction): ...


class IsLastDistinct(BooleanFunction): ...


class IsNan(BooleanFunction): ...


class IsNull(BooleanFunction): ...


class IsUnique(BooleanFunction): ...
