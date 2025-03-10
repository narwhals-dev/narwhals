# mypy: ignore
# ruff: noqa
import re
import subprocess
import sys

out = subprocess.run(["git", "fetch", "upstream", "--tags"])
if out.returncode != 0:
    print(
        "Something went wrong with the release process, please check the Narwhals Wiki and try again."
    )
    print(out)
    sys.exit(1)
subprocess.run(["git", "reset", "--hard", "upstream/main"])

if (
    subprocess.run(
        ["git", "branch", "--show-current"], text=True, capture_output=True
    ).stdout.strip()
    != "bump-version"
):
    msg = "`bump_version.py` should be run from `bump-version` branch"
    raise RuntimeError(msg)

# Delete local tags, if present
try:
    # Get the list of all tags
    result = subprocess.run(
        ["git", "tag", "-l"], capture_output=True, text=True, check=True
    )
    tags = result.stdout.splitlines()  # Split the tags into a list by lines

    # Delete each tag using git tag -d
    for tag in tags:
        subprocess.run(["git", "tag", "-d", tag], check=True)
    print("All local tags have been deleted.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")

subprocess.run(["git", "fetch", "upstream", "--tags"])
subprocess.run(["git", "fetch", "upstream", "--prune", "--tags"])

how = sys.argv[1]

with open("pyproject.toml", encoding="utf-8") as f:
    content = f.read()
old_version = re.search(r'version = "(.*)"', content).group(1)
version = old_version.split(".")
if how == "patch":
    version = ".".join(version[:-1] + [str(int(version[-1]) + 1)])
elif how == "minor":
    version = ".".join(version[:-2] + [str(int(version[-2]) + 1), "0"])
elif how == "major":
    version = ".".join([str(int(version[0]) + 1), "0", "0"])
content = content.replace(f'version = "{old_version}"', f'version = "{version}"')
with open("pyproject.toml", "w", encoding="utf-8") as f:
    f.write(content)

with open("narwhals/__init__.py", encoding="utf-8") as f:
    content = f.read()
content = content.replace(
    f'__version__ = "{old_version}"',
    f'__version__ = "{version}"',
)
with open("narwhals/__init__.py", "w", encoding="utf-8") as f:
    f.write(content)

with open("docs/installation.md", encoding="utf-8") as f:
    content = f.read()
content = content.replace(
    f"'{old_version}'",
    f"'{version}'",
)
with open("docs/installation.md", "w", encoding="utf-8") as f:
    f.write(content)

subprocess.run(["git", "commit", "-a", "-m", f"release: Bump version to {version}"])
subprocess.run(["git", "tag", "-a", f"v{version}", "-m", f"v{version}"])
subprocess.run(["git", "push", "upstream", "HEAD", "--follow-tags"])
subprocess.run(["git", "push", "upstream", "HEAD:stable", "-f", "--follow-tags"])
