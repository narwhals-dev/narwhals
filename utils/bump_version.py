# mypy: ignore
# ruff: noqa
import re
import subprocess
import sys

subprocess.run(["git", "fetch", "upstream"])
subprocess.run(["git", "reset", "--hard", "upstream/main"])

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
