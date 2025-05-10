# mypy: ignore
from __future__ import annotations

import subprocess as sp
import sys

GIT = "git"
UV = "uv"
FETCH = "fetch"
PUSH = "push"
COMMIT = "commit"
UPSTREAM = "upstream"
TAGS = "--tags"
TAG = "tag"
VERSION = "version"

out = sp.run([GIT, FETCH, UPSTREAM, TAGS], check=False)
if out.returncode != 0:
    print(
        "Something went wrong with the release process, please check the Narwhals Wiki for "
        "at https://github.com/narwhals-dev/narwhals/wiki#release-process and try again."
    )
    print(out)
    sys.exit(1)
sp.run([GIT, "reset", "--hard", "upstream/main"], check=False)

if (
    current_branch := sp.run(
        [GIT, "branch", "--show-current"], text=True, capture_output=True, check=False
    ).stdout.strip()
) != "bump-version":
    msg = f"`bump_version.py` should be run from `bump-version` branch instead of `{current_branch}`"
    raise RuntimeError(msg)

# Delete local tags, if present
try:
    # Get the list of all tags
    result = sp.run([GIT, TAG, "-l"], capture_output=True, text=True, check=True)
    tags = result.stdout.splitlines()  # Split the tags into a list by lines

    if tags:
        # Delete each tag using git tag -d
        sp.run([GIT, TAG, "-d", *tags], check=True)

    print("All local tags have been deleted.")
except sp.CalledProcessError as e:
    print(f"An error occurred: {e}")

sp.run([GIT, FETCH, UPSTREAM, TAGS], check=False)
sp.run([GIT, FETCH, UPSTREAM, "--prune", TAGS], check=False)

how = sys.argv[1]

new_version = sp.run(
    [UV, VERSION, "--bump", how, "--short"], capture_output=True, text=True, check=False
).stdout

sp.run([GIT, COMMIT, "-a", "-m", f"release: Bump version to {new_version}"], check=False)
sp.run([GIT, TAG, "-a", f"v{new_version}", "-m", f"v{new_version}"], check=False)
sp.run([GIT, PUSH, UPSTREAM, "HEAD", "--follow-tags"], check=False)
sp.run([GIT, PUSH, UPSTREAM, "HEAD:stable", "-f", "--follow-tags"], check=False)
