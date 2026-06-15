# ruff: noqa: S603, S607
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile

dist_path = Path("dist")
wheel_path = next(dist_path.glob("*.whl"))
sdist_path = next(dist_path.glob("*.tar.gz"))


def git_tracked(*pathspecs: str) -> set[str]:
    """Return the set of git-tracked files under the given pathspecs.

    This is the source of truth for what *must* end up in the distributions:
    if a file is tracked under `src/` or `tests/` it has to be shipped.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z", *pathspecs], capture_output=True, text=True, check=True
    )
    return {name for name in result.stdout.split("\0") if name}


tracked_src = git_tracked("src")
tracked_tests = git_tracked("tests")

errors: list[str] = []

with ZipFile(wheel_path) as wheel_file:
    wheel_names = set(wheel_file.namelist())

    # Allow only 'narwhals' and 'narwhals-<version>.dist-info' (metadata).
    unexpected_wheel_dirs = {
        dir_name
        for name in wheel_names
        if not (dir_name := name.split("/")[0]).startswith("narwhals")
    }
    if unexpected_wheel_dirs:
        errors.append(f"Unexpected directories in wheel: {unexpected_wheel_dirs}")

    # Every tracked source file must ship in the wheel: `src/narwhals/X` -> `narwhals/X`.
    missing_wheel = {
        name for name in tracked_src if name.removeprefix("src/") not in wheel_names
    }
    if missing_wheel:
        errors.append(f"Source files missing from wheel: {sorted(missing_wheel)}")

with TarFile.open(sdist_path, mode="r:gz") as sdist_file:
    # Members are prefixed with 'narwhals-<version>/'; strip that leading component.
    sdist_names = {
        rest
        for m in sdist_file.getmembers()
        if m.isfile() and len(parts := m.name.split("/", 1)) == 2 and (rest := parts[1])
    }

    # Top-level entries that must always be present (source dirs + metadata).
    required_sdist_top_level = {
        "src",
        "tests",
        "pyproject.toml",
        "PKG-INFO",
        "LICENSE.md",
        "README.md",
    }

    sdist_top_level = {name.split("/")[0] for name in sdist_names}
    if unexpected := sdist_top_level - required_sdist_top_level:
        errors.append(f"Unexpected top-level entries in sdist: {unexpected}")
    if missing := required_sdist_top_level - sdist_top_level:
        errors.append(f"Missing required top-level entries in sdist: {missing}")

    # Every tracked file under src/ and tests/ must ship in the sdist.
    # This is the check that guards against regressions like incomplete glob patterns
    # dropping `tests/data/` or other nested directories (see issue #3664).
    if missing_sdist := (tracked_src | tracked_tests) - sdist_names:
        errors.append(f"Tracked files missing from sdist: {sorted(missing_sdist)}")

if errors:
    for error in errors:
        print(f"🚨 {error}")
    sys.exit(1)

print("✅ Distribution content looks good.")
sys.exit(0)
