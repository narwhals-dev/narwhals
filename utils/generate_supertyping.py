"""Adapted from [@FBruzzesi script (2026-01-11)].

[@FBruzzesi script (2026-01-11)]: https://github.com/narwhals-dev/narwhals/pull/3396#issuecomment-3733465005
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Final, TypeVar

import polars as pl

from narwhals.dtypes import DType, Enum, Unknown
from narwhals.dtypes._supertyping import get_supertype

T = TypeVar("T")
DESTINATION_PATH: Final[Path] = Path("docs") / "concepts" / "promotion-rules.md"


def get_leaf_subclasses(cls: type[T]) -> list[type[T]]:
    """Get all leaf subclasses (classes with no further subclasses)."""
    leaves = []
    for subclass in cls.__subclasses__():
        if subclass.__subclasses__():  # Has children, recurse
            leaves.extend(get_leaf_subclasses(subclass))
        else:  # No children, it's a "leaf"
            leaves.append(subclass)
    return leaves


def collect_supertypes() -> None:
    from narwhals.dtypes import _classes as _classes, _classes_v1 as _classes_v1  # noqa: I001, PLC0414

    dtypes = get_leaf_subclasses(DType)
    supertypes: list[tuple[str, str, str]] = []
    for left, right in product(dtypes, dtypes):
        promoted: str
        base_types = frozenset((left, right))
        left_str, right_str = str(left), str(right)

        if Unknown in base_types:
            promoted = str(Unknown)
        elif left is right:
            promoted = str(left)
        elif left.is_nested() or right.is_nested():
            promoted = ""
        else:
            if left is Enum:
                left = Enum(["tmp"])  # noqa: PLW2901
            if right is Enum:
                right = Enum(["tmp"])  # noqa: PLW2901

            _promoted = get_supertype(left(), right())
            promoted = str(_promoted.__class__) if _promoted else ""

        supertypes.append((left_str, right_str, promoted))

    frame = (
        pl.DataFrame(supertypes, schema=["_", "right", "supertype"], orient="row")
        .pivot(
            index="_",
            on="right",
            values="supertype",
            aggregate_function=None,
            sort_columns=True,
        )
        .sort("_")
        .rename({"_": ""})
    )

    with (
        pl.Config(
            tbl_rows=30,
            tbl_cols=30,
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_cell_alignment="LEFT",
            tbl_width_chars=-1,
        ),
        DESTINATION_PATH.open(mode="w", encoding="utf-8", newline="\n") as file,
    ):
        file.write(str(frame))
        file.write("\n")


if __name__ == "__main__":
    collect_supertypes()
