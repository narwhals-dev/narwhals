"""Write redirect stubs into `site/` for docs pages which have moved.

zensical has no equivalent of `mkdocs-redirects` yet (see https://github.com/zensical/backlog/issues/23).
"""

from __future__ import annotations

import sys
from pathlib import Path

SITE = Path("site")
REDIRECTS = {
    "basics/dataframe_conversion": "how-to/dataframe_conversion",
    "concepts/improve_group_by_operation": "how-to/improve_group_by_operation",
    "generating_sql": "how-to/generating_sql",
}
"""Old to new URL, both relative to the site root and without `index.html`."""

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Redirecting</title>
    <link rel="canonical" href="{target}">
    <meta name="robots" content="noindex">
    <meta http-equiv="refresh" content="0; url={target}">
    <script>window.location.replace("{target}" + window.location.hash);</script>
  </head>
  <body>
    <p>Redirecting to <a href="{target}">{target}</a>.</p>
  </body>
</html>
"""


def main() -> int:
    """Write one redirect stub per entry in `REDIRECTS`.

    Returns:
        1 if the site isn't built, a target is missing, or an old path is still a real
        page, else 0.
    """
    if not SITE.is_dir():
        print(f"{SITE} not found: run `zensical build` first.")
        return 1

    for old, new in REDIRECTS.items():
        if not (SITE / new / "index.html").is_file():
            print(f"Redirect target {new}/ does not exist in {SITE}.")
            return 1
        stub = SITE / old / "index.html"
        if stub.is_file():
            print(f"{old}/ is still a real page: drop it from REDIRECTS or move it.")
            return 1
        stub.parent.mkdir(parents=True, exist_ok=True)
        # Relative to the stub, so the site works under any deploy prefix.
        target = f"{'../' * (old.count('/') + 1)}{new}/"
        stub.write_text(TEMPLATE.format(target=target), "utf-8")
        print(f"{old}/ -> {new}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
