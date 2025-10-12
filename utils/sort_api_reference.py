"""Script to automatically sort members lists in API reference markdown files."""
# ruff: noqa: T201

from __future__ import annotations

import re
import sys
from pathlib import Path


def sort_members_in_markdown(file_path: Path) -> int:
    """Sort members lists in a markdown file alphabetically.

    Returns:
        1 if the file was modified, 0 if no changes were needed.
    """
    content = file_path.read_text(encoding="utf-8")

    # Pattern matches "members:" followed by list items (lines starting with "- ")
    pattern = r"(members:)((?:\n\s+-[^\n]+)+)"

    def sort_list(match: re.Match[str]) -> str:
        """Sort a matched members list section."""
        prefix = match.group(1)  # "members:"
        items = match.group(2)  # All list items with newlines and indentation

        item_pattern = r"(\n\s+-)([^\n]+)"  # Extract individual items: (\n + spaces + -) and (content)
        matches = re.findall(item_pattern, items)

        # Sort alphabetically by the item content (excluding indentation and dash)
        sorted_items = sorted(matches, key=lambda x: x[1].strip())

        # Reconstruct: "members:" + sorted items with their indentation preserved
        return prefix + "".join(f"{indent}{content}" for indent, content in sorted_items)

    sorted_content = re.sub(pattern, sort_list, content)
    if sorted_content == content:
        return 0

    file_path.write_text(sorted_content, encoding="utf-8")
    print(f"Sorting members in {file_path}")
    return 1


PATH = Path("docs") / "api-reference"
FILES_TO_SKIP = {"dtypes", "typing"}

ret = max(
    sort_members_in_markdown(file_path=file_path)
    for file_path in PATH.glob("*.md")
    if file_path.stem not in FILES_TO_SKIP
)

sys.exit(ret)
