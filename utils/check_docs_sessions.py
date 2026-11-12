"""Check that `markdown_exec` session names aren't reused across docs pages."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

DOCS = Path("docs")
SESSION_PATTERN = re.compile(r'^\s*```+[^\n`]*\bsession="(?P<name>[^"]+)"')


def find_sessions(path: Path) -> set[str]:
    """Collect the session names used by fenced code blocks in `path`.

    Arguments:
        path: Markdown file to scan.

    Returns:
        Every session name found, deduplicated.
    """
    return {
        match["name"]
        for line in path.read_text("utf-8").splitlines()
        if (match := SESSION_PATTERN.match(line))
    }


def main() -> int:
    """Report session names shared by more than one page.

    Returns:
        1 if any session name is reused across pages, else 0.
    """
    pages: defaultdict[str, list[str]] = defaultdict(list)
    for path in sorted(DOCS.rglob("*.md")):
        for name in sorted(find_sessions(path)):
            pages[name].append(path.as_posix())

    clashes = {name: paths for name, paths in pages.items() if len(paths) > 1}
    if not clashes:
        return 0

    for name, paths in clashes.items():
        print(f'Session name "{name}" is used by more than one page:')
        for path in paths:
            print(f"  {path}")
    print(
        "\n`markdown_exec` sessions share globals across the whole build, so session names "
        "must be unique per page. Rename all but one of the pages above."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
