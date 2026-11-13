"""Bump a package's version, then commit, tag and push it. Pushing the tag publishes to PyPI.

See RELEASING.md
"""

from __future__ import annotations

import argparse
import os
import subprocess as sp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

BumpKind = Literal["patch", "minor", "major"]

GIT = "git"
UV = "uv"
REMOTE = "upstream"


@dataclass(frozen=True)
class Package:
    name: str
    directory: Path
    branch: str
    tag_prefix: str
    commit_subject: str  # Followed by the new version.


PACKAGES = {
    p.name: p
    for p in (
        Package("narwhals", Path(), "bump-version", "v", "release: Bump version to"),
        Package(
            "narwhals-dask",
            Path("packages") / "narwhals-dask",
            "bump-narwhals-dask",
            "narwhals-dask-v",
            "release: Bump narwhals-dask version to",
        ),
    )
}


def run(*args: str) -> str:
    try:
        return sp.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except sp.CalledProcessError as exc:
        sys.exit(f"`{' '.join(args)}` failed:\n{exc.stderr.strip()}")


def bump_lock_entry(lock: Path, package: str, old: str, new: str) -> None:
    old_entry = f'name = "{package}"\nversion = "{old}"\n'.encode()
    new_entry = f'name = "{package}"\nversion = "{new}"\n'.encode()
    content = lock.read_bytes()
    if (count := content.count(old_entry)) != 1:
        sys.exit(f"Expected one {package} {old} entry in {lock}, found {count}.")
    lock.write_bytes(content.replace(old_entry, new_entry))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "bump",
        choices=get_args(BumpKind),
        help="Part of the version to bump (major.minor.patch).",
    )
    parser.add_argument(
        "--package", choices=PACKAGES, default="narwhals", help="Package to release."
    )
    args = parser.parse_args()
    bump: BumpKind = args.bump
    pkg = PACKAGES[args.package]

    pyproject = str(pkg.directory / "pyproject.toml")
    lock = str(pkg.directory / "uv.lock")
    uv_version = (UV, "version", "--directory", str(pkg.directory))

    os.chdir(Path(__file__).resolve().parents[1])
    if run(GIT, "status", "--porcelain", "--untracked-files=no"):
        sys.exit("Commit or stash your changes to tracked files first.")

    run(GIT, "fetch", REMOTE, "--prune", "--prune-tags", "--force")
    run(GIT, "switch", "-C", pkg.branch, f"{REMOTE}/main")

    old_version = run(*uv_version, "--short")
    new_version = run(*uv_version, "--bump", bump, "--short", "--dry-run", "--frozen")
    if not new_version:
        sys.exit("`uv version --bump` returned an empty version.")
    tag = f"{pkg.tag_prefix}{new_version}"
    if run(GIT, "tag", "--list", tag):
        sys.exit(f"Tag {tag} already exists on {REMOTE}.")

    try:
        # `uv lock` re-spells every fork's resolution markers and Renovate's
        # `uv lock --upgrade` reverts them, so edit only the package's entry.
        run(*uv_version, "--bump", bump, "--frozen")
        bump_lock_entry(Path(lock), pkg.name, old_version, new_version)
        run(UV, "lock", "--check", "--directory", str(pkg.directory))

        run(GIT, "add", pyproject, lock)
        staged = run(GIT, "diff", "--cached", "--numstat").splitlines()
        if staged != [f"1\t1\t{pyproject}", f"1\t1\t{lock}"]:
            sys.exit(
                f"Expected a one-line change in {pyproject} and {lock}, got:\n"
                + "\n".join(staged)
            )
        run(GIT, "commit", "-m", f"{pkg.commit_subject} {new_version}")
        print(run(GIT, "show", "--stat", "HEAD"))

        answer = input(
            f"\nTag and push {tag} to {REMOTE}? This publishes {pkg.name} to PyPI. [y/N] "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            print(f"Nothing pushed. The next run resets {pkg.branch}.")
            return

        run(GIT, "tag", "-a", tag, "-m", tag)
        run(
            GIT,
            "push",
            "--atomic",
            REMOTE,
            f"HEAD:refs/heads/{pkg.branch}",
            f"refs/tags/{tag}",
        )
    except BaseException:
        # The tree was clean before the bump, so this only undoes the script's changes.
        sp.run([GIT, "restore", "--staged", "--worktree", pyproject, lock], check=False)
        sp.run([GIT, "tag", "--delete", tag], capture_output=True, check=False)
        raise

    print(
        f"Pushed {pkg.branch} and {tag}. Open a PR from {pkg.branch} and squash-merge it.\n"
        "Do not amend, force-push or re-tag: the tag and the PyPI release use this commit."
    )


if __name__ == "__main__":
    main()
