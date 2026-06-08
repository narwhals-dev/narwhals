"""Hoping to add better support for [PEP 681].

See [Supporting custom decorators], [extensions/dataclasses.py], [griffe_pydantic]

Run with:

    uvx griffe dump --extensions utils/griffe_extension.py src/narwhals

[PEP 681]: https://peps.python.org/pep-0681/
[Supporting custom decorators]: https://mkdocstrings.github.io/griffe/guide/users/how-to/support-decorators/
[extensions/dataclasses.py]: https://github.com/mkdocstrings/griffe/blob/5dc97b78f318ac97016fbaee83a574ff15516f58/packages/griffelib/src/griffe/_internal/extensions/dataclasses.py
[griffe_pydantic]: https://github.com/mkdocstrings/griffe-pydantic/blob/9df87609b4edc2e548d15691e4d6c26710861daa/src/griffe_pydantic/_internal/extension.py
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import griffe

if TYPE_CHECKING:
    import logging


if TYPE_CHECKING:
    logger = logging.getLogger(__name__)
else:
    # NOTE: griff's logger has a `__getattr__`
    logger = griffe.get_logger(__name__)


GriffeAny: TypeAlias = griffe.Object | griffe.Expr


class PEP681Extension(griffe.Extension):
    """An extension to support `@dataclass_transform`.

    Important:
        Worked backwords from [`extensions/dataclasses.py`](https://github.com/mkdocstrings/griffe/blob/5dc97b78f318ac97016fbaee83a574ff15516f58/packages/griffelib/src/griffe/_internal/extensions/dataclasses.py)
    """

    def __init__(self) -> None:
        super().__init__()

    def on_package(
        self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **kwargs: Any
    ) -> None:
        path = pkg.canonical_path
        logger.info("Starting: %r", path)
        _apply_recursively(pkg, set())
        logger.info("Finished: %r", path)


CANONICAL_PATH_PLAN = "narwhals._plan"
CANONICAL_PATH_IMMUTABLE = f"{CANONICAL_PATH_PLAN}._immutable.Immutable"


def _apply_recursively(
    mod_cls: griffe.Module | griffe.Class, processed: set[str]
) -> None:
    """`griffe._internal.extensions.dataclasses.__apply_recursively`.

    `processed` means **seen**.
    """
    if mod_cls.canonical_path in processed:
        return
    processed.add(mod_cls.canonical_path)
    if isinstance(mod_cls, griffe.Class):
        if "__init__" not in mod_cls.members:
            _set_dataclass_init(mod_cls)
        for member in mod_cls.members.values():
            if not member.is_alias and member.is_class:
                _apply_recursively(member, processed)  # pyright: ignore[reportArgumentType]
    elif isinstance(mod_cls, griffe.Module):
        for member in mod_cls.members.values():
            if not member.is_alias and (member.is_module or member.is_class):
                _apply_recursively(member, processed)  # pyright: ignore[reportArgumentType]


def relative_path(cls: GriffeAny) -> str:
    """Strip the `narwhals._plan` prefix, for logging."""
    return _relative_path(cls.canonical_path)


@cache
def _relative_path(path: str) -> str:
    r = path.removeprefix(CANONICAL_PATH_PLAN)
    if r == path:
        msg = f"Expected all classes that reach here to start with {CANONICAL_PATH_PLAN!r}\nGot: {path!r}"
        raise NotImplementedError(msg)
    return r


def _set_dataclass_init(class_: griffe.Class) -> None:
    """`griffe._internal.extensions.dataclasses._set_dataclass_init`."""
    # Retrieve parameters from all parent dataclasses.
    parameters: list[griffe.Parameter] = []
    try:
        mro = class_.mro()
    except ValueError:
        mro = ()
    for parent in reversed(mro):
        if inherits_immutable(parent):
            parameters.extend(_dataclass_parameters(parent))
            class_.labels.add("dataclass")

    if not inherits_immutable(class_):
        return

    logger.info("Handling `Immutable` child: %r", relative_path(class_))

    # Add current class parameters.
    parameters.extend(_dataclass_parameters(class_))
    if not parameters:
        logger.info("|   Skipped (no params)")
        return

    logger.info(
        f"|   Parameters(*, {', '.join(f'{p.name}: {p.annotation}' for p in parameters)})"  # noqa: G004
    )

    # Create `__init__` method with re-ordered parameters.
    init = griffe.Function(
        "__init__",
        lineno=0,
        endlineno=0,
        parent=class_,
        parameters=griffe.Parameters(
            griffe.Parameter(
                name="self",
                annotation=None,
                kind=griffe.ParameterKind.positional_or_keyword,
                default=None,
            ),
            *_reorder_parameters(parameters),
        ),
        returns="None",
    )
    class_.set_member("__init__", init)
    logger.info(
        f"|   {init.signature(name=class_.name + '.__init__')}"  # noqa: G004
    )


@cache
def inherits_immutable(cls: griffe.Class) -> bool:
    # like `_dataclass_decorator` / `_inherits_pydantic`
    if any(canonical_path(base) == CANONICAL_PATH_IMMUTABLE for base in cls.bases):
        return True
    return any(
        inherits_immutable(parent)
        for parent in cls.mro()
        if parent.canonical_path.startswith(CANONICAL_PATH_PLAN)
    )


def canonical_path(obj: str | griffe.Expr) -> str:
    return obj.canonical_path if isinstance(obj, griffe.Expr) else obj


# TODO @dangotbanned: Fix `ClassVar` broken for:
#   - `IRDateTimeNamespace`
#   - `IRStringNamespace`
#   - `IRListNamespace`
#   - `IRCatNamespace`
#   - `__function_parameters__: ClassVar`
# TODO @dangotbanned: `ExprIR` is showing `node(s)` as a default
# TODO @dangotbanned: Skip adding an `__init__` when there are no parameters
#   - Loads of functions like that
@cache
def _dataclass_parameters(class_: griffe.Class) -> list[griffe.Parameter]:
    # Iterate on current attributes to find parameters.
    parameters = []
    for member in class_.members.values():
        if member.is_attribute:
            member = cast("griffe.Attribute", member)

            # All dataclass parameters have annotations.
            if member.annotation is None:
                continue

            # Attributes that have labels for these characteristics are
            # not class parameters:
            # - @property
            # - @cached_property
            # - ClassVar annotation
            if "property" in member.labels or (
                # TODO: It is better to explicitly check for `ClassVar`, but  # noqa: TD002
                # `Visitor.handle_attribute` unwraps it from the annotation.
                # Maybe create `internal_labels` and store "classvar" in there.
                "class-attribute" in member.labels
                and "instance-attribute" not in member.labels
            ):
                continue

            # Add parameter to the list.
            parameters.append(
                griffe.Parameter(
                    member.name,
                    annotation=member.annotation,
                    # All parameters marked as keyword-only.
                    kind=griffe.ParameterKind.keyword_only,
                    default=member.value,
                    docstring=member.docstring,
                )
            )

    return parameters


def _reorder_parameters(parameters: list[griffe.Parameter]) -> list[griffe.Parameter]:
    # De-duplicate, overwriting previous parameters.
    params_dict = {param.name: param for param in parameters}

    # Re-order, putting positional-only in front and keyword-only at the end.
    pos_only = []
    pos_kw = []
    kw_only = []
    for param in params_dict.values():
        if param.kind is griffe.ParameterKind.positional_only:
            pos_only.append(param)
        elif param.kind is griffe.ParameterKind.keyword_only:
            kw_only.append(param)
        else:
            pos_kw.append(param)
    return pos_only + pos_kw + kw_only
