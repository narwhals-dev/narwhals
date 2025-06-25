"""Backward compatibility for newer/less buggy typing features.

## Important
Import from here to avoid introducing a runtime dependency on [`typing_extensions`]

## Notes
- `Protocol38`
  - https://github.com/narwhals-dev/narwhals/pull/2064#discussion_r1965921386
  - https://github.com/narwhals-dev/narwhals/pull/2294#discussion_r2014534830
- `TypeVar` defaults
  - https://typing.python.org/en/latest/spec/generics.html#type-parameter-defaults
  - https://peps.python.org/pep-0696/
- `@deprecated`
  - https://docs.python.org/3/library/warnings.html#warnings.deprecated
  - https://typing.python.org/en/latest/spec/directives.html#deprecated
  - https://peps.python.org/pep-0702/

[`typing_extensions`]: https://github.com/python/typing_extensions
"""

from __future__ import annotations

# ruff: noqa: ARG001, ANN202, N802
import sys
from typing import TYPE_CHECKING, Any


# TODO @dangotbanned: `version` validation
# - For static typing, `message` must be a `LiteralString`
# - So the `narwhals` version needs to be embedded in the string, without using fstrings/str.format/etc
# - We'll need to decide on a style to use, and then add **runtime** validation to ensure we stay conistent
#   - E.g. "<thing> is deprecated since narwhals <version>. Use <alternative> instead. <Extended description>"
#   - Where only the <alternative> and <Extended description> sections are optional.
def _deprecated_compat(
    message: str, /, *, category: type[DeprecationWarning] | None = DeprecationWarning
) -> Callable[[Callable[P, R]], Callable[P, R]]:  # pragma: no cover
    def decorate(func: Callable[P, R], /) -> Callable[P, R]:
        if category is None:
            func.__deprecated__ = message  # type: ignore[attr-defined]
            return func

        # TODO @dangotbanned: Coverage for this before `3.13`?
        if isinstance(func, type) or not callable(func):  # pragma: no cover
            from narwhals._utils import qualified_type_name

            # NOTE: The logic for that part is much more complex, leaving support out *for now*,
            # as we don't have any deprecated classes.
            # https://github.com/python/cpython/blob/eec7a8ff22dcf409717a21a9aeab28b55526ee24/Lib/_py_warnings.py#L745-L789
            msg = f"@nw._typing_compat.deprecated` cannot be applied to {qualified_type_name(func)!r}"
            raise NotImplementedError(msg)
        cat = category
        import functools

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwds: P.kwargs) -> R:
            from narwhals._utils import issue_deprecation_warning

            issue_deprecation_warning(message, _version="???", category=cat)
            return func(*args, **kwds)

        return wrapper

    return decorate


if TYPE_CHECKING:
    from typing import Callable, Protocol as Protocol38

    from typing_extensions import ParamSpec

    if sys.version_info >= (3, 13):
        from typing import TypeVar
        from warnings import deprecated
    else:
        from typing_extensions import TypeVar, deprecated

    if sys.version_info >= (3, 11):
        from typing import Never, assert_never
    else:
        from typing_extensions import Never, assert_never

    P = ParamSpec("P")
    R = TypeVar("R")


else:  # pragma: no cover
    if sys.version_info >= (3, 13):
        from typing import TypeVar
        from warnings import deprecated
    else:
        from typing import TypeVar as _TypeVar

        def TypeVar(
            name: str,
            *constraints: Any,
            bound: Any | None = None,
            covariant: bool = False,
            contravariant: bool = False,
            **kwds: Any,
        ):
            return _TypeVar(
                name,
                *constraints,
                bound=bound,
                covariant=covariant,
                contravariant=contravariant,
            )

        deprecated = _deprecated_compat

    _ASSERT_NEVER_REPR_MAX_LENGTH = 100
    _BUG_URL = (
        "https://github.com/narwhals-dev/narwhals/issues/new?template=bug_report.yml"
    )

    def assert_never(arg: Never, /) -> Never:
        value = repr(arg)
        if len(value) > _ASSERT_NEVER_REPR_MAX_LENGTH:
            value = value[:_ASSERT_NEVER_REPR_MAX_LENGTH] + "..."
        msg = (
            f"Expected code to be unreachable, but got: {value}.\n"
            f"Please report an issue at {_BUG_URL}"
        )
        raise AssertionError(msg)

    # TODO @dangotbanned: Remove after dropping `3.8` (#2084)
    # - https://github.com/narwhals-dev/narwhals/pull/2064#discussion_r1965921386
    from typing import Protocol as Protocol38


__all__ = ["Protocol38", "TypeVar", "assert_never", "deprecated"]
