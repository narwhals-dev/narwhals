# Contributing

Thank you for your interest in contributing to Narwhals! Any kind of improvement is welcome!

## Setting up your environment

Here's how you can set up your local development environment to contribute:

1. Make sure you have Python3.8+ installed (for example, Python 3.11)
2. Create a new virtual environment with `python3.11 -m venv .venv` (or whichever version of Python3.9+ you prefer)
3. Activate it: `. .venv/bin/activate`
4. Install Narwhals: `pip install -e .`
5. Install test requirements: `pip install -r requirements-dev.txt`
6. Install docs requirements: `pip install -r docs/requirements-docs.txt`

You should also install pre-commit:
```
pip install pre-commit
pre-commit install
```
This will automatically format and lint your code before each commit, and it will block the commit if any issues are found.

## Working on your issue

Create a new git branch from the `main` branch in your local repository.
Note that your work cannot be merged if the test below fail.
If you add code that should be tested, please add tests.

## Running tests

To run tests, run `pytest`. To check coverage: `pytest --cov=narwhals`.
To run tests on the docset-module, use `pytest narwhals --doctest-modules`.

If you want to have less surprises when opening a PR, you can take advantage of [nox](https://nox.thea.codes/en/stable/index.html) to run the entire CI/CD test suite locally in your operating system.

To do so, you will first need to install nox and then run the `nox` command in the root of the repository:

```bash
python -m pip install nox  # python -m pip install "nox[uv]"
nox
```

Notice that nox will also require to have all the python versions that are defined in the `noxfile.py` installed in your system.

## Building docs

To build the docs, run `mkdocs serve`, and then open the link provided in a browser.
The docs should refresh when you make changes. If they don't, press `ctrl+C`, and then
do `mkdocs build` and then `mkdocs serve`.

## Pull requests

When you have resolved your issue, [open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) in the Narwhals repository.

Please adhere to the following guidelines:

1. Start your pull request title with a [conventional commit](https://www.conventionalcommits.org/) tag. This helps us add your contribution to the right section of the changelog. We use "Type" from the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#type).
2. Use a descriptive title starting with an uppercase letter. This text will end up in the [changelog](https://github.com/narwhals-dev/narwhals/releases).
3. Please follow the instructions in the pull request form and submit. 

## How it works

If Narwhals looks like underwater unicorn magic to you, then please read
[how it works](https://narwhals-dev.github.io/narwhals/how-it-works/).

## Happy contributing!

Please remember to abide by the code of conduct, else you'll be conducted away from this project.

## Community Calendar

We have a community call every 2 weeks, all welcome to attend.

[Subscribe to our calendar](https://calendar.google.com/calendar/embed?src=27ff6dc5f598c1d94c1f6e627a1aaae680e2fac88f848bda1f2c7946ae74d5ab%40group.calendar.google.com).
