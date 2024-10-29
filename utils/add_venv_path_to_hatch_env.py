# ruff: noqa
from __future__ import annotations

import toml


def add_venv_path_to_hatch_env(
    pyproject_file="pyproject.toml", env_name="default", venv_path="./.venv"
):
    try:
        # Load the pyproject.toml file
        with open(pyproject_file) as file:
            data = toml.load(file)

        # Navigate to the Hatch environment section
        hatch_envs = (
            data.setdefault("tool", {}).setdefault("hatch", {}).setdefault("envs", {})
        )

        # Access the specific environment (default or specified by env_name)
        env = hatch_envs.setdefault(env_name, {})

        # Check if 'path' is already set, add if missing
        if "path" not in env:
            env["path"] = venv_path
            print(f"Added 'path = \"{venv_path}\"' to [tool.hatch.envs.{env_name}]")
        else:
            print(f"'path' is already set in [tool.hatch.envs.{env_name}]")

        # Write the modified data back to the pyproject.toml file
        with open(pyproject_file, "w") as file:
            toml.dump(data, file)
        print(f"Updated {pyproject_file} successfully.")

    except FileNotFoundError:
        print(f"Error: {pyproject_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Run the function to modify pyproject.toml
add_venv_path_to_hatch_env()
